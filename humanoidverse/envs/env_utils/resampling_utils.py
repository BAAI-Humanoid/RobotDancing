import torch


def compute_segment_layout(motion_lengths, segment_duration_s):
    if segment_duration_s <= 0:
        raise ValueError("segment_duration_s must be positive")

    motion_lengths = torch.as_tensor(motion_lengths, dtype=torch.float32)
    segment_counts = torch.clamp(
        torch.ceil(motion_lengths / float(segment_duration_s)).long(),
        min=1,
    )
    max_segments = int(segment_counts.max().item())
    segment_ids = torch.arange(max_segments, device=motion_lengths.device)
    valid_mask = segment_ids.unsqueeze(0) < segment_counts.unsqueeze(1)
    return segment_counts, valid_mask


def compute_max_start_times(motion_lengths, frame_dts):
    return torch.clamp(
        torch.as_tensor(motion_lengths) - torch.as_tensor(frame_dts),
        min=0.0,
    )


def segment_time_bounds(
    segment_indices,
    motion_lengths,
    frame_dts,
    segment_duration_s,
    max_time_ratios=None,
):
    max_start_times = compute_max_start_times(motion_lengths, frame_dts)
    if max_time_ratios is not None:
        max_start_times = torch.minimum(
            max_start_times,
            torch.as_tensor(motion_lengths) * torch.as_tensor(max_time_ratios),
        )
    starts = segment_indices.to(dtype=torch.float32) * float(segment_duration_s)
    ends = torch.minimum(starts + float(segment_duration_s), max_start_times)
    return starts, torch.maximum(ends, starts), max_start_times


def mask_segments_by_max_time(valid_mask, segment_duration_s, max_start_times):
    segment_starts = (
        torch.arange(valid_mask.shape[1], device=valid_mask.device)
        .to(dtype=torch.float32)
        * float(segment_duration_s)
    )
    constrained = valid_mask & (
        segment_starts.unsqueeze(0) <= max_start_times.unsqueeze(1)
    )
    has_segment = constrained.any(dim=1, keepdim=True)
    return torch.where(has_segment, constrained, valid_mask)


def masked_uniform(valid_mask):
    weights = valid_mask.to(dtype=torch.float32)
    return weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)


def normalize_masked_scores(scores, valid_mask):
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.clamp(scores, min=0.0) * valid_mask
    sums = scores.sum(dim=1, keepdim=True)
    uniform = masked_uniform(valid_mask)
    return torch.where(
        sums > 0,
        scores / torch.clamp(sums, min=1e-12),
        uniform,
    )


def apply_distribution_floor(weights, valid_mask, min_uniform_ratio):
    if not 0.0 <= min_uniform_ratio <= 1.0:
        raise ValueError("distribution_min_uniform_ratio must be in [0, 1]")

    weights = torch.clamp(weights, min=0.0) * valid_mask
    sums = weights.sum(dim=1, keepdim=True)
    uniform = masked_uniform(valid_mask)
    normalized = torch.where(
        sums > 0,
        weights / torch.clamp(sums, min=1e-12),
        uniform,
    )
    floored = (
        (1.0 - min_uniform_ratio) * normalized
        + min_uniform_ratio * uniform
    )
    required_floor = min_uniform_ratio * uniform
    already_valid = (
        (~valid_mask)
        | (normalized >= required_floor - 1e-7)
    ).all(dim=1, keepdim=True)
    return torch.where(already_valid, normalized, floored)


def failure_ramp(common_step_counter, warmup_steps, ramp_steps):
    if common_step_counter < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(
        max((common_step_counter - warmup_steps) / float(ramp_steps), 0.0),
        1.0,
    )


def combined_mixture_weights(
    distribution_ramp_value,
    failure_ramp_value,
    handoff_value,
    initial_distribution_weight,
    initial_failure_weight,
    final_distribution_weight=None,
    final_failure_weight=None,
):
    """Return staged uniform/distribution/failure mixture weights."""
    values = (
        distribution_ramp_value,
        failure_ramp_value,
        handoff_value,
    )
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Mixture ramp coefficients must be in [0, 1]")
    if final_distribution_weight is None:
        final_distribution_weight = initial_distribution_weight
    if final_failure_weight is None:
        final_failure_weight = initial_failure_weight

    initial_sum = initial_distribution_weight + initial_failure_weight
    final_sum = final_distribution_weight + final_failure_weight
    if any(
        not 0.0 <= value <= 1.0
        for value in (
            initial_distribution_weight,
            initial_failure_weight,
            final_distribution_weight,
            final_failure_weight,
        )
    ):
        raise ValueError("Combined mixture weights must be in [0, 1]")
    if initial_sum > 1.0 or final_sum > 1.0:
        raise ValueError(
            "Combined distribution and failure weights must sum to at most 1"
        )

    distribution_target = (
        (1.0 - handoff_value) * initial_distribution_weight
        + handoff_value * final_distribution_weight
    )
    failure_target = (
        (1.0 - handoff_value) * initial_failure_weight
        + handoff_value * final_failure_weight
    )
    distribution_weight = distribution_ramp_value * distribution_target
    failure_weight = failure_ramp_value * failure_target
    uniform_weight = 1.0 - distribution_weight - failure_weight
    if uniform_weight < -1e-8:
        raise ValueError("Active combined mixture weights exceed one")
    return max(uniform_weight, 0.0), distribution_weight, failure_weight


def eligible_episode_mask(active_episode_mask, episode_lengths):
    return active_episode_mask.bool() & (episode_lengths > 0)


def should_update_failure_statistics(
    common_step_counter,
    last_update_step,
    update_interval_steps,
    force=False,
):
    if update_interval_steps <= 0:
        raise ValueError("failure_stats_update_interval_steps must be positive")
    return force or (
        common_step_counter - last_update_step >= update_interval_steps
    )


def shrink_failure_rates(
    failure_rates,
    exposure_counts,
    failure_counts,
    valid_mask,
    min_samples_for_stats,
    confidence_exposure,
):
    if min_samples_for_stats < 0:
        raise ValueError("min_samples_for_stats must be non-negative")
    if confidence_exposure <= 0:
        raise ValueError("failure_confidence_exposure must be positive")

    valid_float = valid_mask.to(dtype=torch.float32)
    total_exposures = (exposure_counts.float() * valid_float).sum(
        dim=1, keepdim=True
    )
    total_failures = (failure_counts.float() * valid_float).sum(
        dim=1, keepdim=True
    )
    global_rates = total_failures / torch.clamp(total_exposures, min=1.0)

    excess_exposure = torch.clamp(
        exposure_counts.float() - float(min_samples_for_stats), min=0.0
    )
    confidence = excess_exposure / (
        excess_exposure + float(confidence_exposure)
    )
    confidence = confidence * valid_float
    shrunk_rates = (
        confidence * torch.clamp(failure_rates, min=0.0, max=1.0)
        + (1.0 - confidence) * global_rates
    )
    return shrunk_rates * valid_float, confidence


def accumulate_predecessor_failure_credit(
    motion_indices,
    terminal_segments,
    failed_mask,
    num_motions,
    max_segments,
    lookback_segments,
    decay_time_segments,
):
    if lookback_segments < 0:
        raise ValueError("lookback_segments must be non-negative")
    if decay_time_segments <= 0:
        raise ValueError("decay_time_segments must be positive")
    if not (
        motion_indices.shape == terminal_segments.shape == failed_mask.shape
    ):
        raise ValueError("Failure-credit inputs must have identical shapes")

    credit = torch.zeros(
        (num_motions, max_segments),
        dtype=torch.float32,
        device=motion_indices.device,
    )
    failed_indices = torch.nonzero(failed_mask.bool(), as_tuple=False).flatten()
    if failed_indices.numel() == 0:
        return credit

    offsets = torch.arange(
        lookback_segments + 1,
        dtype=torch.long,
        device=motion_indices.device,
    )
    weights = torch.exp(
        -offsets.float() / float(decay_time_segments)
    )
    motions = motion_indices[failed_indices].long().unsqueeze(1)
    terminals = terminal_segments[failed_indices].long().unsqueeze(1)
    segments = terminals - offsets.unsqueeze(0)
    valid = segments >= 0
    flat_indices = (
        motions.expand_as(segments) * max_segments
        + torch.clamp(segments, min=0)
    )
    credit.view(-1).scatter_add_(
        0,
        flat_indices[valid],
        weights.unsqueeze(0).expand_as(segments)[valid],
    )
    return credit


def select_hard_windows_nms(
    scores,
    valid_mask,
    top_k,
    suppression_radius,
    minimum_selected=1,
):
    if top_k <= 0:
        raise ValueError("hard_window_top_k must be positive")
    if suppression_radius < 0:
        raise ValueError("suppression_radius must be non-negative")
    if minimum_selected <= 0:
        raise ValueError("minimum_selected must be positive")

    selected = torch.zeros_like(valid_mask)
    for motion_idx in range(scores.shape[0]):
        candidates = torch.nonzero(
            valid_mask[motion_idx], as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            continue
        order = candidates[
            torch.argsort(scores[motion_idx, candidates], descending=True)
        ]
        chosen = []
        for segment in order.tolist():
            if all(
                abs(segment - previous) > suppression_radius
                for previous in chosen
            ):
                chosen.append(segment)
                if len(chosen) >= top_k:
                    break
        if len(chosen) < min(top_k, minimum_selected):
            for segment in order.tolist():
                if segment not in chosen:
                    chosen.append(segment)
                    if len(chosen) >= min(top_k, minimum_selected):
                        break
        if chosen:
            selected[motion_idx, chosen] = True
    return selected


def select_earliest_frontier_windows(
    failure_rates,
    confidence,
    scores,
    valid_mask,
    top_k,
    failure_threshold,
    confidence_threshold,
    span_segments,
    suppression_radius,
    minimum_segment=0,
    excluded_mask=None,
    preferred_mask=None,
    allow_fallback=False,
    contiguous_selection=False,
):
    if not 0.0 <= failure_threshold <= 1.0:
        raise ValueError("frontier_failure_threshold must be in [0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("frontier_confidence_threshold must be in [0, 1]")
    if span_segments <= 0:
        raise ValueError("frontier_span_segments must be positive")

    frontier_mask = torch.zeros_like(valid_mask)
    anchors = torch.full(
        (valid_mask.shape[0],),
        -1,
        dtype=torch.long,
        device=valid_mask.device,
    )
    segment_ids = torch.arange(
        valid_mask.shape[1], device=valid_mask.device
    )
    if excluded_mask is None:
        excluded_mask = torch.zeros_like(valid_mask)
    if preferred_mask is None:
        preferred_mask = torch.zeros_like(valid_mask)
    for motion_idx in range(valid_mask.shape[0]):
        if torch.is_tensor(minimum_segment):
            row_minimum_segment = int(
                minimum_segment[motion_idx].item()
            )
        else:
            row_minimum_segment = int(minimum_segment)
        eligible = (
            valid_mask[motion_idx]
            & ~excluded_mask[motion_idx]
            & (segment_ids >= row_minimum_segment)
            & (failure_rates[motion_idx] >= failure_threshold)
            & (confidence[motion_idx] >= confidence_threshold)
        )
        candidates = torch.nonzero(eligible, as_tuple=False).flatten()
        if candidates.numel() == 0:
            if not allow_fallback:
                continue
            fallback = (
                valid_mask[motion_idx]
                & ~excluded_mask[motion_idx]
                & (segment_ids >= row_minimum_segment)
            )
            fallback_indices = torch.nonzero(
                fallback, as_tuple=False
            ).flatten()
            if fallback_indices.numel() == 0:
                continue
            anchor = fallback_indices[
                torch.argmax(scores[motion_idx, fallback_indices])
            ]
        else:
            anchor = candidates.min()
        anchors[motion_idx] = anchor
        band = (
            valid_mask[motion_idx]
            & ~excluded_mask[motion_idx]
            & (segment_ids >= anchor)
            & (segment_ids < anchor + span_segments)
        )
        if contiguous_selection:
            chosen = []
            for segment in torch.nonzero(
                band, as_tuple=False
            ).flatten().tolist():
                if all(
                    abs(segment - previous) > suppression_radius
                    for previous in chosen
                ):
                    chosen.append(segment)
                    if len(chosen) >= top_k:
                        break
            if chosen:
                frontier_mask[motion_idx, chosen] = True
            continue
        selection_scores = scores[motion_idx : motion_idx + 1].clone()
        selection_scores[0, preferred_mask[motion_idx] & band] += (
            torch.clamp(selection_scores.max(), min=1.0) * 1000.0
        )
        selected = select_hard_windows_nms(
            selection_scores,
            band.unsqueeze(0),
            top_k,
            suppression_radius,
            minimum_selected=min(top_k, int(band.sum().item())),
        )
        frontier_mask[motion_idx] = selected[0]
    return frontier_mask, anchors


def classify_frontier_targets(
    frontier_mask,
    attempts,
    successes,
    min_attempts,
    mastered_threshold,
    impossible_threshold,
):
    if min_attempts <= 0:
        raise ValueError("frontier_min_attempts must be positive")
    if not 0.0 <= impossible_threshold < mastered_threshold <= 1.0:
        raise ValueError(
            "Frontier thresholds must satisfy "
            "0 <= impossible < mastered <= 1"
        )
    rates = successes.float() / torch.clamp(attempts.float(), min=1.0)
    enough = frontier_mask & (attempts >= min_attempts)
    mastered = enough & (rates >= mastered_threshold)
    impossible = enough & (rates <= impossible_threshold)
    learnable = frontier_mask & ~(mastered | impossible)
    return rates, mastered, impossible, learnable


def frontier_crossing_success(
    target_segments,
    terminal_segments,
    failed_mask,
    segment_counts,
    crossing_margin_segments,
):
    if crossing_margin_segments <= 0:
        raise ValueError("crossing_margin_segments must be positive")
    if not (
        target_segments.shape
        == terminal_segments.shape
        == failed_mask.shape
        == segment_counts.shape
    ):
        raise ValueError("Frontier crossing inputs must have identical shapes")
    last_segments = segment_counts.long() - 1
    required_segments = torch.minimum(
        target_segments.long() + int(crossing_margin_segments),
        last_segments,
    )
    crossed = terminal_segments.long() >= required_segments
    reaches_motion_end = required_segments >= last_segments
    return crossed & (~reaches_motion_end | ~failed_mask.bool())


def normalize_sparse_hard_window_probabilities(
    scores,
    selected_mask,
    valid_mask,
    hard_window_ratio,
    max_probability_ratio,
    iterations=50,
):
    if not 0.0 < hard_window_ratio <= 1.0:
        raise ValueError("hard_window_ratio must be in (0, 1]")
    if max_probability_ratio < 1.0:
        raise ValueError(
            "hard_window_max_probability_ratio must be at least 1"
        )
    selected_float = selected_mask.to(dtype=torch.float32)
    selected_counts = selected_float.sum(dim=1, keepdim=True)
    segment_counts = valid_mask.float().sum(dim=1, keepdim=True)
    conditional_upper = torch.minimum(
        torch.ones_like(segment_counts),
        max_probability_ratio
        / torch.clamp(segment_counts * hard_window_ratio, min=1.0),
    )
    # Near the end of a motion fewer candidates may remain than the configured
    # cap can support. Relax only enough to keep the sparse distribution
    # feasible instead of failing the training run.
    conditional_upper = torch.maximum(
        conditional_upper,
        1.0 / torch.clamp(selected_counts, min=1.0),
    )

    normalized = normalize_masked_scores(scores, selected_mask)
    low_scale = torch.zeros_like(conditional_upper)
    high_scale = torch.ones_like(conditional_upper)
    for _ in range(20):
        projected = torch.minimum(
            normalized * high_scale, conditional_upper
        ) * selected_float
        high_scale = torch.where(
            projected.sum(dim=1, keepdim=True) < 1.0,
            high_scale * 2.0,
            high_scale,
        )
    for _ in range(iterations):
        scale = 0.5 * (low_scale + high_scale)
        projected = torch.minimum(
            normalized * scale, conditional_upper
        ) * selected_float
        needs_more = projected.sum(dim=1, keepdim=True) < 1.0
        low_scale = torch.where(needs_more, scale, low_scale)
        high_scale = torch.where(needs_more, high_scale, scale)
    projected = torch.minimum(
        normalized * high_scale, conditional_upper
    ) * selected_float
    return projected / torch.clamp(
        projected.sum(dim=1, keepdim=True), min=1e-12
    )


def compute_hard_window_scores(
    failure_rates,
    predecessor_credit,
    learning_progress,
    immediate_failure_rates,
    valid_mask,
    tracking_errors=None,
    tracking_error_scale=0.35,
    tracking_quality_floor=0.2,
    failure_regression=None,
    failure_regression_weight=0.0,
):
    if tracking_error_scale <= 0:
        raise ValueError("tracking_error_scale must be positive")
    if not 0.0 <= tracking_quality_floor <= 1.0:
        raise ValueError("tracking_quality_floor must be in [0, 1]")
    frontier = torch.clamp(failure_rates, 0.0, 1.0)
    frontier = frontier * (1.0 - frontier)
    predecessor = normalize_masked_scores(predecessor_credit, valid_mask)
    progress = normalize_masked_scores(
        torch.clamp(learning_progress, min=0.0), valid_mask
    )
    if failure_regression is None:
        regression = torch.zeros_like(failure_rates)
    else:
        regression = normalize_masked_scores(
            torch.clamp(failure_regression, min=0.0), valid_mask
        )
    if failure_regression_weight < 0.0:
        raise ValueError("failure_regression_weight must be non-negative")
    frontier = normalize_masked_scores(frontier, valid_mask)
    viability = torch.where(
        immediate_failure_rates >= 0.95,
        torch.zeros_like(immediate_failure_rates),
        torch.clamp(1.0 - immediate_failure_rates, min=0.05, max=1.0),
    )
    if tracking_errors is None:
        tracking_quality = torch.ones_like(failure_rates)
    else:
        tracking_quality = torch.exp(
            -torch.clamp(tracking_errors, min=0.0)
            / float(tracking_error_scale)
        )
        tracking_quality = torch.clamp(
            tracking_quality, min=tracking_quality_floor, max=1.0
        )
    scores = (
        0.45 * frontier + 0.35 * predecessor + 0.20 * progress
        + failure_regression_weight * regression
    ) * viability * tracking_quality
    return scores * valid_mask


def suppress_immediate_failure_probabilities(
    probabilities,
    immediate_failure_rates,
    immediate_exposures,
    valid_mask,
    threshold,
    min_exposures=32,
    suppression_floor=0.1,
):
    """Downweight direct resets into reliably non-viable terminal segments."""
    if not 0.0 <= threshold < 1.0:
        raise ValueError("Immediate-failure threshold must be in [0, 1)")
    if min_exposures < 0:
        raise ValueError("Immediate-failure min exposures must be non-negative")
    if not 0.0 <= suppression_floor <= 1.0:
        raise ValueError("Immediate-failure suppression floor must be in [0, 1]")

    reliable = (immediate_exposures >= min_exposures) & valid_mask
    excess = torch.clamp(
        (immediate_failure_rates - threshold) / max(1.0 - threshold, 1e-6),
        min=0.0,
        max=1.0,
    )
    multiplier = 1.0 - excess * (1.0 - suppression_floor)
    multiplier = torch.where(reliable, multiplier, torch.ones_like(multiplier))
    adjusted = probabilities * multiplier * valid_mask
    return normalize_masked_scores(adjusted, valid_mask), multiplier


def hard_window_start_times(
    target_segments,
    lookback_seconds,
    jitter_seconds,
    segment_duration_s,
    max_start_times,
    random_values,
):
    if jitter_seconds < 0:
        raise ValueError("hard_window_jitter_s must be non-negative")
    target_times = target_segments.float() * float(segment_duration_s)
    starts = target_times - lookback_seconds.float()
    jitter = (2.0 * random_values.float() - 1.0) * float(jitter_seconds)
    return torch.minimum(
        torch.clamp(starts + jitter, min=0.0),
        max_start_times,
    )


def sampling_pool_masks(random_values, hard_ratio, distribution_ratio):
    if hard_ratio < 0 or distribution_ratio < 0:
        raise ValueError("Sampling pool ratios must be non-negative")
    if hard_ratio + distribution_ratio > 1.0:
        raise ValueError("Sampling pool ratios must sum to at most 1")
    hard = random_values < hard_ratio
    distribution = (
        (random_values >= hard_ratio)
        & (random_values < hard_ratio + distribution_ratio)
    )
    return hard, distribution, ~(hard | distribution)


def contextual_sampling_pool_masks(
    random_values,
    frontier_ratio,
    hard_ratio,
    distribution_ratio,
):
    ratios = (frontier_ratio, hard_ratio, distribution_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Sampling pool ratios must be non-negative")
    if sum(ratios) > 1.0:
        raise ValueError("Sampling pool ratios must sum to at most 1")
    frontier = random_values < frontier_ratio
    hard = (
        (random_values >= frontier_ratio)
        & (random_values < frontier_ratio + hard_ratio)
    )
    distribution = (
        (random_values >= frontier_ratio + hard_ratio)
        & (
            random_values
            < frontier_ratio + hard_ratio + distribution_ratio
        )
    )
    return frontier, hard, distribution, ~(
        frontier | hard | distribution
    )


def dominant_contextual_sampling_pool_masks(
    random_values,
    frontier_ratio,
    hard_ratio,
    distribution_ratio,
    failure_ratio,
):
    ratios = (frontier_ratio, hard_ratio, distribution_ratio, failure_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Sampling pool ratios must be non-negative")
    if sum(ratios) > 1.0:
        raise ValueError("Sampling pool ratios must sum to at most 1")
    frontier = random_values < frontier_ratio
    hard = (
        (random_values >= frontier_ratio)
        & (random_values < frontier_ratio + hard_ratio)
    )
    distribution = (
        (random_values >= frontier_ratio + hard_ratio)
        & (
            random_values
            < frontier_ratio + hard_ratio + distribution_ratio
        )
    )
    failure = (
        (random_values >= frontier_ratio + hard_ratio + distribution_ratio)
        & (
            random_values
            < frontier_ratio
            + hard_ratio
            + distribution_ratio
            + failure_ratio
        )
    )
    return frontier, hard, distribution, failure, ~(
        frontier | hard | distribution | failure
    )


def distribution_failure_topk_conflict(
    distribution_prior,
    failure_prior,
    valid_mask,
    topk_fraction=0.1,
):
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")
    valid_counts = valid_mask.sum(dim=1).long()
    max_topk = int(
        torch.clamp(
            torch.ceil(valid_counts.float().max() * topk_fraction),
            min=1,
        ).item()
    )
    row_topk = torch.clamp(
        torch.ceil(valid_counts.float() * topk_fraction).long(),
        min=1,
    )
    ranks = torch.arange(max_topk, device=valid_mask.device).unsqueeze(0)
    selected = ranks < row_topk.unsqueeze(1)

    dist_scores = distribution_prior.masked_fill(~valid_mask, -float("inf"))
    failure_scores = failure_prior.masked_fill(~valid_mask, -float("inf"))
    _, dist_indices = torch.topk(dist_scores, k=max_topk, dim=1)
    _, failure_indices = torch.topk(failure_scores, k=max_topk, dim=1)
    dist_mask = torch.zeros_like(valid_mask)
    failure_mask = torch.zeros_like(valid_mask)
    dist_mask.scatter_(1, dist_indices, selected)
    failure_mask.scatter_(1, failure_indices, selected)
    overlap = (dist_mask & failure_mask & valid_mask).sum(dim=1).float()
    topk = torch.clamp(row_topk.float(), min=1.0)
    return 1.0 - overlap / topk


def update_hard_window_lookbacks(
    current_lookbacks,
    immediate_failure_rates,
    immediate_exposures,
    valid_mask,
    threshold,
    increment_s,
    max_lookback_s,
    min_exposures=32,
):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Immediate-failure threshold must be in [0, 1]")
    needs_more_context = (
        (immediate_exposures >= min_exposures)
        & (immediate_failure_rates > threshold)
        & valid_mask
    )
    return torch.where(
        needs_more_context,
        torch.clamp(current_lookbacks + increment_s, max=max_lookback_s),
        current_lookbacks,
    )


def project_probabilities_with_bounds(
    probabilities,
    valid_mask,
    min_uniform_ratio,
    max_probability_ratio,
    iterations=50,
):
    if not 0.0 <= min_uniform_ratio <= 1.0:
        raise ValueError("uniform_mix_ratio must be in [0, 1]")
    if max_probability_ratio < 1.0:
        raise ValueError("max_sampling_probability_ratio must be at least 1")

    valid_float = valid_mask.to(dtype=probabilities.dtype)
    segment_counts = valid_float.sum(dim=1, keepdim=True)
    lower = min_uniform_ratio / torch.clamp(segment_counts, min=1.0)
    upper = torch.minimum(
        torch.full_like(lower, 1.0),
        max_probability_ratio / torch.clamp(segment_counts, min=1.0),
    )
    if torch.any(lower * segment_counts > 1.0 + 1e-6):
        raise ValueError("Probability lower bounds are infeasible")
    if torch.any(upper * segment_counts < 1.0 - 1e-6):
        raise ValueError("Probability upper bounds are infeasible")

    normalized = torch.clamp(probabilities, min=0.0) * valid_float
    normalized = normalized / torch.clamp(
        normalized.sum(dim=1, keepdim=True), min=1e-12
    )
    cap_hits = ((normalized > upper) & valid_mask).sum(dim=1)

    low_scale = torch.zeros_like(lower)
    high_scale = torch.ones_like(lower)
    for _ in range(20):
        projected = torch.clamp(
            normalized * high_scale, min=lower, max=upper
        ) * valid_float
        needs_more_mass = projected.sum(dim=1, keepdim=True) < 1.0
        high_scale = torch.where(
            needs_more_mass, high_scale * 2.0, high_scale
        )

    for _ in range(iterations):
        scale = 0.5 * (low_scale + high_scale)
        projected = torch.clamp(
            normalized * scale, min=lower, max=upper
        ) * valid_float
        needs_more_mass = projected.sum(dim=1, keepdim=True) < 1.0
        low_scale = torch.where(needs_more_mass, scale, low_scale)
        high_scale = torch.where(needs_more_mass, high_scale, scale)

    projected = torch.clamp(
        normalized * high_scale, min=lower, max=upper
    ) * valid_float
    projected = projected / torch.clamp(
        projected.sum(dim=1, keepdim=True), min=1e-12
    )
    return projected, cap_hits


def compute_sampling_probabilities(
    valid_mask,
    distribution_weights,
    failure_rates,
    exposure_counts,
    failure_counts,
    enable_distribution,
    enable_failure,
    min_samples_for_stats,
    failure_confidence_exposure,
    failure_rate_epsilon,
    failure_bias_strength,
    distribution_strength,
    uniform_mix_ratio,
    max_sampling_probability_ratio,
    failure_ramp_value,
    combined_mode="mixture",
    combined_failure_weight=0.4,
    distribution_ramp_value=1.0,
    combined_distribution_weight=0.2,
    failure_priority_mode="frontier",
    combined_handoff_value=0.0,
    combined_distribution_weight_final=None,
    combined_failure_weight_final=None,
):
    if not 0.0 <= uniform_mix_ratio <= 1.0:
        raise ValueError("uniform_mix_ratio must be in [0, 1]")
    if failure_rate_epsilon <= 0:
        raise ValueError("failure_rate_epsilon must be positive")
    if distribution_strength <= 0:
        raise ValueError("distribution_strength must be positive")
    if combined_mode not in ("mixture", "product"):
        raise ValueError("combined_mode must be 'mixture' or 'product'")
    final_distribution_weight = (
        combined_distribution_weight
        if combined_distribution_weight_final is None
        else combined_distribution_weight_final
    )
    final_failure_weight = (
        combined_failure_weight
        if combined_failure_weight_final is None
        else combined_failure_weight_final
    )
    for name, value in (
        ("combined_distribution_weight", combined_distribution_weight),
        ("combined_failure_weight", combined_failure_weight),
        ("combined_distribution_weight_final", final_distribution_weight),
        ("combined_failure_weight_final", final_failure_weight),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if (
        combined_distribution_weight + combined_failure_weight > 1.0
        or final_distribution_weight + final_failure_weight > 1.0
    ):
        raise ValueError(
            "Combined distribution and failure weights must sum to at most 1"
        )
    if not 0.0 <= combined_handoff_value <= 1.0:
        raise ValueError("combined_handoff_value must be in [0, 1]")
    if not 0.0 <= distribution_ramp_value <= 1.0:
        raise ValueError("distribution_ramp_value must be in [0, 1]")
    if failure_priority_mode not in ("raw", "frontier"):
        raise ValueError("failure_priority_mode must be 'raw' or 'frontier'")

    uniform = masked_uniform(valid_mask)
    if enable_distribution:
        distribution_scores = (
            torch.pow(
                torch.clamp(distribution_weights, min=0.0) + 1e-12,
                distribution_strength,
            )
            * valid_mask
        )
        distribution_sums = distribution_scores.sum(dim=1, keepdim=True)
        distribution = torch.where(
            distribution_sums > 0,
            distribution_scores
            / torch.clamp(distribution_sums, min=1e-12),
            uniform,
        )
    else:
        distribution = uniform

    if enable_failure:
        shrunk_rates, confidence = shrink_failure_rates(
            failure_rates,
            exposure_counts,
            failure_counts,
            valid_mask,
            min_samples_for_stats,
            failure_confidence_exposure,
        )
        if failure_priority_mode == "frontier":
            priority_rates = shrunk_rates * (1.0 - shrunk_rates)
        else:
            priority_rates = shrunk_rates
        failure_scores = torch.pow(
            priority_rates + failure_rate_epsilon, failure_bias_strength
        )
        failure_scores = failure_scores * valid_mask
        failure_sums = failure_scores.sum(dim=1, keepdim=True)
        failure_prior = torch.where(
            failure_sums > 0,
            failure_scores / torch.clamp(failure_sums, min=1e-12),
            uniform,
        )
    else:
        failure_prior = uniform
        confidence = torch.zeros_like(failure_rates)

    if enable_distribution and enable_failure:
        if combined_mode == "product":
            ramped_distribution = (
                (1.0 - distribution_ramp_value) * uniform
                + distribution_ramp_value * distribution
            )
            ramped_failure = (
                (1.0 - failure_ramp_value) * uniform
                + failure_ramp_value * failure_prior
            )
            biased = ramped_distribution * ramped_failure
        else:
            (
                uniform_weight,
                distribution_weight,
                failure_weight,
            ) = combined_mixture_weights(
                float(distribution_ramp_value),
                float(failure_ramp_value),
                float(combined_handoff_value),
                float(combined_distribution_weight),
                float(combined_failure_weight),
                combined_distribution_weight_final,
                combined_failure_weight_final,
            )
            biased = (
                uniform_weight * uniform
                + distribution_weight * distribution
                + failure_weight * failure_prior
            )
    elif enable_distribution:
        biased = (
            (1.0 - distribution_ramp_value) * uniform
            + distribution_ramp_value * distribution
        )
    elif enable_failure:
        biased = (
            (1.0 - failure_ramp_value) * uniform
            + failure_ramp_value * failure_prior
        )
    else:
        return uniform, distribution, failure_prior, confidence, torch.zeros(
            valid_mask.shape[0], dtype=torch.long, device=valid_mask.device
        )

    biased = torch.nan_to_num(biased, nan=0.0, posinf=0.0, neginf=0.0)
    biased = torch.clamp(biased, min=0.0) * valid_mask
    biased_sums = biased.sum(dim=1, keepdim=True)
    biased = torch.where(
        biased_sums > 0,
        biased / torch.clamp(biased_sums, min=1e-12),
        uniform,
    )
    if enable_distribution and enable_failure and combined_mode == "mixture":
        final = biased
    else:
        final = (
            uniform_mix_ratio * uniform
            + (1.0 - uniform_mix_ratio) * biased
        )
    final, cap_hits = project_probabilities_with_bounds(
        final,
        valid_mask,
        uniform_mix_ratio,
        max_sampling_probability_ratio,
    )
    return final, distribution, failure_prior, confidence, cap_hits


def _masked_correlation(lhs, rhs, valid_mask):
    valid_float = valid_mask.to(dtype=lhs.dtype)
    counts = torch.clamp(valid_float.sum(dim=1), min=1.0)
    lhs_mean = (lhs * valid_float).sum(dim=1) / counts
    rhs_mean = (rhs * valid_float).sum(dim=1) / counts
    lhs_centered = (lhs - lhs_mean.unsqueeze(1)) * valid_float
    rhs_centered = (rhs - rhs_mean.unsqueeze(1)) * valid_float
    covariance = (lhs_centered * rhs_centered).sum(dim=1)
    lhs_var = torch.square(lhs_centered).sum(dim=1)
    rhs_var = torch.square(rhs_centered).sum(dim=1)
    denom = torch.sqrt(torch.clamp(lhs_var * rhs_var, min=1e-24))
    return covariance / denom


def prior_alignment_diagnostics(
    final_probabilities,
    distribution_prior,
    failure_prior,
    valid_mask,
    high_failure_fraction=0.1,
):
    if not 0.0 < high_failure_fraction <= 1.0:
        raise ValueError("high_failure_fraction must be in (0, 1]")

    final = final_probabilities * valid_mask
    distribution = distribution_prior * valid_mask
    failure = failure_prior * valid_mask
    valid_counts = valid_mask.sum(dim=1).long()
    max_topk = int(
        torch.clamp(
            torch.ceil(valid_counts.float().max() * high_failure_fraction),
            min=1,
        ).item()
    )
    row_topk = torch.clamp(
        torch.ceil(valid_counts.float() * high_failure_fraction).long(),
        min=1,
    )
    failure_scores = failure.masked_fill(~valid_mask, -float("inf"))
    _, topk_indices = torch.topk(failure_scores, k=max_topk, dim=1)
    topk_ranks = torch.arange(max_topk, device=valid_mask.device).unsqueeze(0)
    selected = topk_ranks < row_topk.unsqueeze(1)
    high_failure_mask = torch.zeros_like(valid_mask)
    high_failure_mask.scatter_(1, topk_indices, selected)
    high_failure_mask = high_failure_mask & valid_mask

    high_failure_mass = failure * high_failure_mask
    suppressed = high_failure_mask & (final < failure)
    suppressed_count = suppressed.sum()
    gate_loss_per_motion = (
        torch.clamp(failure - final, min=0.0) * high_failure_mask
    ).sum(dim=1) / torch.clamp(high_failure_mass.sum(dim=1), min=1e-12)

    return {
        "final_distribution_corr": _masked_correlation(
            final, distribution, valid_mask
        ).mean(),
        "final_failure_corr": _masked_correlation(
            final, failure, valid_mask
        ).mean(),
        "suppressed_high_failure_count": suppressed_count.float(),
        "distribution_gate_loss": gate_loss_per_motion.mean(),
    }


def accumulate_episode_statistics(
    motion_indices,
    start_segments,
    terminal_segments,
    failed_mask,
    num_motions,
    max_segments,
):
    device = motion_indices.device
    motion_indices = motion_indices.long()
    start_segments = start_segments.long()
    terminal_segments = terminal_segments.long()
    failed_mask = failed_mask.bool()

    if not (
        motion_indices.shape
        == start_segments.shape
        == terminal_segments.shape
        == failed_mask.shape
    ):
        raise ValueError("Episode statistic inputs must have identical shapes")

    difference_width = max_segments + 1
    difference = torch.zeros(
        num_motions * difference_width, dtype=torch.long, device=device
    )
    start_indices = motion_indices * difference_width + start_segments
    end_indices = (
        motion_indices * difference_width
        + torch.clamp(terminal_segments + 1, max=max_segments)
    )
    difference.scatter_add_(0, start_indices, torch.ones_like(start_indices))
    difference.scatter_add_(0, end_indices, -torch.ones_like(end_indices))
    exposures = torch.cumsum(
        difference.view(num_motions, difference_width), dim=1
    )[:, :max_segments]

    failures = torch.zeros(
        num_motions * max_segments, dtype=torch.long, device=device
    )
    failed_indices = (
        motion_indices[failed_mask] * max_segments
        + terminal_segments[failed_mask]
    )
    failures.scatter_add_(
        0, failed_indices, torch.ones_like(failed_indices)
    )
    return exposures, failures.view(num_motions, max_segments)


def update_failure_ema(
    failure_rates,
    batch_exposures,
    batch_failures,
    alpha,
):
    if not 0.0 < alpha <= 1.0:
        raise ValueError("failure_rate_ema_alpha must be in (0, 1]")

    updated = failure_rates.clone()
    observed = batch_exposures > 0
    batch_rates = batch_failures.float() / torch.clamp(
        batch_exposures.float(), min=1.0
    )
    updated[observed] = (
        (1.0 - alpha) * updated[observed] + alpha * batch_rates[observed]
    )
    return updated


def sampling_summary(probabilities, valid_mask, previous_probabilities=None):
    masked_probs = probabilities * valid_mask
    uniform = masked_uniform(valid_mask)
    entropy_per_motion = -(
        masked_probs * torch.log(torch.clamp(masked_probs, min=1e-12))
    ).sum(dim=1)
    effective_sample_size_per_motion = 1.0 / torch.clamp(
        torch.square(masked_probs).sum(dim=1), min=1e-12
    )
    valid_probs = probabilities[valid_mask]
    summary = {
        "entropy": entropy_per_motion.mean(),
        "effective_sample_size": effective_sample_size_per_motion.mean(),
        "max_probability": valid_probs.max(),
        "min_probability": valid_probs.min(),
        "uniform_kl": (
            masked_probs
            * (
                torch.log(torch.clamp(masked_probs, min=1e-12))
                - torch.log(torch.clamp(uniform, min=1e-12))
            )
        ).sum(dim=1).mean(),
    }
    if previous_probabilities is None:
        summary["probability_total_variation"] = torch.zeros(
            (), dtype=probabilities.dtype, device=probabilities.device
        )
        summary["probability_kl_to_previous"] = torch.zeros(
            (), dtype=probabilities.dtype, device=probabilities.device
        )
    else:
        previous = previous_probabilities * valid_mask
        summary["probability_total_variation"] = (
            0.5 * torch.abs(masked_probs - previous).sum(dim=1)
        ).mean()
        summary["probability_kl_to_previous"] = (
            masked_probs
            * (
                torch.log(torch.clamp(masked_probs, min=1e-12))
                - torch.log(torch.clamp(previous, min=1e-12))
            )
        ).sum(dim=1).mean()
    return summary
