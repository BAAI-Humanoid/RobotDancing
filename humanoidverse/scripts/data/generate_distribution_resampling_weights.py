#!/usr/bin/env python3
"""Generate one-second distribution-aware segment weights for resampling."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", required=True, help="Motion pkl file or directory of pkl files.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--segment-duration-s", type=float, default=1.0)
    parser.add_argument(
        "--num-time-segments",
        type=int,
        default=None,
        help="Legacy fixed-count segmentation. Ignored when --segment-duration-s is set.",
    )
    parser.add_argument("--distribution-min-uniform-ratio", type=float, default=0.1)
    parser.add_argument("--joint-indices", type=int, nargs="+", default=[0, 3, 6, 9])
    parser.add_argument("--num-bins-per-joint", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def load_motion_entries(path):
    import joblib

    path = Path(path)
    if path.is_dir():
        pkl_files = sorted(path.glob("*.pkl"))
    else:
        pkl_files = [path]

    entries = {}
    for pkl_file in pkl_files:
        data = joblib.load(pkl_file)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for key, value in data.items():
                entries[str(key)] = value
        elif isinstance(data, dict):
            entries[pkl_file.stem] = data
        else:
            raise ValueError(f"Unsupported motion data format in {pkl_file}")
    if not entries:
        raise ValueError(f"No motion entries found in {path}")
    return entries


def extract_joint_positions(motion, joint_indices):
    if "dof" in motion:
        dof = np.asarray(motion["dof"], dtype=np.float64)
    elif "dof_pos" in motion:
        dof = np.asarray(motion["dof_pos"], dtype=np.float64)
    elif "pose_aa" in motion:
        pose_aa = np.asarray(motion["pose_aa"], dtype=np.float64)
        dof = pose_aa.reshape(pose_aa.shape[0], -1)
    else:
        raise KeyError("Motion entry must contain one of: dof, dof_pos, pose_aa")

    if dof.ndim != 2:
        dof = dof.reshape(dof.shape[0], -1)
    max_index = max(joint_indices)
    if max_index >= dof.shape[1]:
        raise IndexError(f"joint index {max_index} is out of bounds for motion DOF dimension {dof.shape[1]}")
    return dof[:, joint_indices]


def motion_fps(motion):
    fps = float(motion.get("fps", 0.0))
    if fps <= 0:
        raise ValueError("Motion entry must contain a positive fps")
    return fps


def split_segments(values, fps, segment_duration_s):
    num_frames = values.shape[0]
    duration = max(num_frames - 1, 0) / fps
    num_segments = max(int(np.ceil(duration / segment_duration_s)), 1)
    frame_segment_ids = np.floor(
        np.arange(num_frames, dtype=np.float64) / fps / segment_duration_s
    ).astype(np.int64)
    frame_segment_ids = np.clip(frame_segment_ids, 0, num_segments - 1)
    segments = []
    for segment_idx in range(num_segments):
        segment_values = values[frame_segment_ids == segment_idx]
        if len(segment_values) == 0:
            nearest_frame = min(int(round(segment_idx * segment_duration_s * fps)), num_frames - 1)
            segment_values = values[nearest_frame:nearest_frame + 1]
        segments.append(segment_values)
    return segments


def segment_histograms(motion_values, motion_fps_by_key, segment_duration_s, num_bins, ranges):
    histograms = []
    segment_ids = []
    for motion_key, values in motion_values.items():
        segments = split_segments(
            values, motion_fps_by_key[motion_key], segment_duration_s
        )
        for segment_idx, segment_values in enumerate(segments):
            hist, _ = np.histogramdd(segment_values, bins=num_bins, range=ranges)
            hist = hist.astype(np.float64).reshape(-1)
            total = hist.sum()
            if total > 0:
                hist /= total
            histograms.append(hist)
            segment_ids.append((motion_key, segment_idx))
    return np.stack(histograms, axis=1), segment_ids


def project_to_simplex(values):
    """Euclidean projection onto {x | x >= 0, sum(x) = 1}."""
    if values.size == 1:
        return np.ones_like(values)
    sorted_values = np.sort(values)[::-1]
    cumsum = np.cumsum(sorted_values)
    rho_candidates = sorted_values * np.arange(1, values.size + 1) > (cumsum - 1.0)
    if not np.any(rho_candidates):
        return np.ones_like(values) / values.size
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cumsum[rho] - 1.0) / float(rho + 1)
    return np.maximum(values - theta, 0.0)


def solve_segment_weights(histograms, iterations):
    target = np.ones(histograms.shape[0], dtype=np.float64) / histograms.shape[0]
    weights = np.ones(histograms.shape[1], dtype=np.float64) / histograms.shape[1]

    # Frobenius norm is an inexpensive upper bound on the spectral norm, which
    # gives a conservative stable step size without an SVD on large segment sets.
    frobenius_norm = np.linalg.norm(histograms, ord="fro")
    step_size = 1.0 / (frobenius_norm * frobenius_norm + 1e-12)

    for _ in range(iterations):
        residual = histograms @ weights - target
        gradient = histograms.T @ residual
        weights = project_to_simplex(weights - step_size * gradient)
    return weights


def normalize_motion_segment_weights(global_weights, segment_ids, min_uniform_ratio):
    grouped = {}
    for weight, (motion_key, segment_idx) in zip(global_weights, segment_ids):
        if motion_key not in grouped:
            count = sum(key == motion_key for key, _ in segment_ids)
            grouped[motion_key] = np.zeros(count, dtype=np.float64)
        grouped[motion_key][segment_idx] = weight

    for motion_key, weights in grouped.items():
        num_segments = len(weights)
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            normalized = np.ones(num_segments, dtype=np.float64) / num_segments
        else:
            normalized = weights / total
        uniform = np.ones(num_segments, dtype=np.float64) / num_segments
        grouped[motion_key] = (
            (1.0 - min_uniform_ratio) * normalized
            + min_uniform_ratio * uniform
        )
    return grouped


def main():
    args = parse_args()
    if args.segment_duration_s <= 0:
        raise ValueError("--segment-duration-s must be positive")
    if not 0.0 <= args.distribution_min_uniform_ratio <= 1.0:
        raise ValueError("--distribution-min-uniform-ratio must be in [0, 1]")
    if args.num_bins_per_joint <= 1:
        raise ValueError("--num-bins-per-joint must be greater than 1")

    entries = load_motion_entries(args.motion_file)
    motion_values = {
        key: extract_joint_positions(motion, args.joint_indices)
        for key, motion in entries.items()
    }
    motion_fps_by_key = {key: motion_fps(motion) for key, motion in entries.items()}

    all_values = np.concatenate(list(motion_values.values()), axis=0)
    mins = all_values.min(axis=0)
    maxs = all_values.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    ranges = [(float(lo - 1e-6 * span), float(hi + 1e-6 * span)) for lo, hi, span in zip(mins, maxs, spans)]

    histograms, segment_ids = segment_histograms(
        motion_values,
        motion_fps_by_key,
        args.segment_duration_s,
        args.num_bins_per_joint,
        ranges,
    )
    global_weights = solve_segment_weights(histograms, args.iterations)
    motion_weights = normalize_motion_segment_weights(
        global_weights,
        segment_ids,
        args.distribution_min_uniform_ratio,
    )
    motion_segment_counts = {
        key: len(weights) for key, weights in motion_weights.items()
    }

    output = {
        "schema_version": 2,
        "method": "simplex_projected_least_squares_histogram_balancing",
        "segment_duration_s": args.segment_duration_s,
        "motion_segment_counts": motion_segment_counts,
        "distribution_min_uniform_ratio": args.distribution_min_uniform_ratio,
        "joint_indices": args.joint_indices,
        "num_bins_per_joint": args.num_bins_per_joint,
        "source_motion_file": str(args.motion_file),
        "motions": {key: weights.tolist() for key, weights in motion_weights.items()},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
