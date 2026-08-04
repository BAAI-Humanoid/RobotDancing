#!/usr/bin/env python3
"""Launch common fixed-start revision evaluation for one or more checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shlex
from pathlib import Path

import yaml

try:
    from .experiment_utils import (
        parse_gpu_ids,
        run_experiments,
        write_manifest,
    )
except ImportError:
    from experiment_utils import (
        parse_gpu_ids,
        run_experiments,
        write_manifest,
    )


def _resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_motion_payload(path):
    try:
        import joblib
    except ModuleNotFoundError:
        joblib = None

    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as stream:
        return pickle.load(stream)


def _motion_entries(path):
    path = Path(path)
    if path.is_dir():
        pkl_files = sorted(path.glob("*.pkl"))
    else:
        pkl_files = [path]
    entries = {}
    for pkl_file in pkl_files:
        payload = _load_motion_payload(pkl_file)
        if (
            isinstance(payload, dict)
            and payload
            and all(isinstance(value, dict) for value in payload.values())
        ):
            entries.update({str(key): value for key, value in payload.items()})
        elif isinstance(payload, dict):
            entries[pkl_file.stem] = payload
        else:
            raise ValueError(f"Unsupported motion data format: {pkl_file}")
    if not entries:
        raise ValueError(f"No motion entries found in {path}")
    return entries


def _motion_duration_and_frame_dt(motion):
    fps = float(motion.get("fps", 0.0))
    if fps <= 0:
        raise ValueError("Motion entry must contain a positive fps")
    for key in ("dof", "dof_pos", "pose_aa", "root_trans_offset"):
        values = motion.get(key)
        if values is not None:
            num_frames = int(
                values.shape[0] if hasattr(values, "shape") else len(values)
            )
            break
    else:
        raise KeyError(
            "Motion entry must contain one of: dof, dof_pos, pose_aa, "
            "root_trans_offset"
        )
    duration = max(num_frames - 1, 0) / fps
    return duration, 1.0 / fps


def infer_rollout_count_from_motion_file(
    motion_file,
    start_interval_s=1.0,
    start_offset_s=0.0,
):
    if start_interval_s <= 0:
        raise ValueError("start_interval_s must be positive")
    entries = _motion_entries(motion_file)
    if len(entries) != 1:
        raise ValueError(
            "Automatic fixed-start count currently requires one motion; "
            "set --rollouts explicitly for multi-motion evaluation"
        )
    motion = next(iter(entries.values()))
    duration, frame_dt = _motion_duration_and_frame_dt(motion)
    max_start_time = max(duration - frame_dt, 0.0)
    if start_offset_s > max_start_time:
        return 1
    inferred_count = (
        math.floor((max_start_time - start_offset_s) / start_interval_s)
        + 1
    )
    return max(
        int(inferred_count),
        1,
    )


def infer_rollout_count(
    checkpoint,
    start_interval_s=1.0,
    start_offset_s=0.0,
):
    run_dir = checkpoint.parent
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot infer fixed-start count without run config: {run_dir}"
        )
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    motion_config = config["robot"]["motion"]
    weights_value = config["robot"]["motion"].get(
        "distribution_weights_path"
    )
    if not weights_value:
        motion_file = motion_config.get("motion_file")
        if not motion_file:
            raise ValueError(
                "Set --rollouts explicitly because the run config has "
                "neither distribution weight metadata nor a motion_file"
            )
        return infer_rollout_count_from_motion_file(
            _resolve_path(motion_file),
            start_interval_s=start_interval_s,
            start_offset_s=start_offset_s,
        )
    weights_path = _resolve_path(weights_value)
    with open(weights_path, encoding="utf-8") as stream:
        weights = json.load(stream)
    counts = list(weights.get("motion_segment_counts", {}).values())
    if len(counts) != 1:
        raise ValueError(
            "Automatic fixed-start count currently requires one motion; "
            "set --rollouts explicitly for multi-motion evaluation"
        )
    return int(counts[0])


def parse_iterations(value):
    if not value:
        return ()
    return tuple(
        int(item.strip()) for item in value.split(",") if item.strip()
    )


def expand_checkpoints(checkpoint_values, iterations):
    checkpoints = []
    for checkpoint_value in checkpoint_values:
        checkpoint = Path(checkpoint_value).resolve()
        run_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
        if iterations:
            for iteration in iterations:
                candidate = run_dir / f"model_{iteration}.pt"
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                checkpoints.append(candidate)
        else:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoints.append(checkpoint)
    return list(dict.fromkeys(checkpoints))


def expand_manifest_checkpoints(
    manifest_path,
    log_root,
    iterations,
    allow_missing,
):
    with open(manifest_path, encoding="utf-8") as stream:
        training_records = json.load(stream)
    checkpoints = []
    identities = {}
    missing = []
    for record in training_records:
        experiment_name = record["experiment_name"]
        final_iteration = int(record["iterations"])
        candidates = sorted(
            path
            for path in Path(log_root).glob(f"*-{experiment_name}-*")
            if (path / f"model_{final_iteration}.pt").is_file()
        )
        if not candidates:
            missing.append(experiment_name)
            continue
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple completed runs found for {experiment_name}: "
                + ", ".join(str(path) for path in candidates)
            )
        run_dir = candidates[0]
        requested = iterations or (final_iteration,)
        for iteration in requested:
            checkpoint = (run_dir / f"model_{iteration}.pt").resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoints.append(checkpoint)
            identities[str(checkpoint)] = {
                (
                    "training_experiment_name"
                    if key == "experiment_name"
                    else key
                ): record[key]
                for key in (
                    "experiment_name",
                    "study",
                    "motion",
                    "condition",
                    "sampling",
                    "platform",
                    "motion_file",
                    "sampler_version",
                    "protocol_id",
                    "num_envs",
                    "seed",
                    "baseline",
                    "algorithm",
                    "obs_config",
                    "reward_config",
                    "use_vec_reward",
                )
                if key in record
            }
    if missing and not allow_missing:
        raise FileNotFoundError(
            "Completed final checkpoints not found for: "
            + ", ".join(missing)
        )
    if missing:
        print(
            f"Skipping {len(missing)} incomplete training run(s): "
            + ", ".join(missing),
            flush=True,
        )
    return checkpoints, identities


def checkpoint_iteration(checkpoint):
    try:
        return int(checkpoint.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def write_progression_summary(output_root, records):
    rows = []
    for record in records:
        rollouts_path = Path(record["output_dir"]) / "rollouts.csv"
        if not rollouts_path.is_file():
            continue
        with open(rollouts_path, newline="", encoding="utf-8") as stream:
            rollouts = list(csv.DictReader(stream))
        if not rollouts:
            continue
        completion = [
            int(row["completed"]) for row in rollouts
        ]
        survival = [
            float(row["survival_time_s"]) for row in rollouts
        ]
        rows.append(
            {
                "run": record["run_name"],
                "checkpoint": record["checkpoint"],
                "iteration": record["iteration"],
                "protocol": record["protocol"],
                "rollout_count": len(rollouts),
                "completion_rate": sum(completion) / len(completion),
                "mean_survival_time_s": sum(survival) / len(survival),
                "min_survival_time_s": min(survival),
                "max_survival_time_s": max(survival),
                "first_failure_segment": rollouts[0].get(
                    "failure_segment", ""
                )
                if record["protocol"] == "full_sequence"
                else "",
            }
        )
    if not rows:
        return
    csv_path = output_root / "checkpoint_progression.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(
        output_root / "checkpoint_progression.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*")
    parser.add_argument(
        "--training-manifest",
        help=(
            "Training manifest whose completed runs should be evaluated. "
            "Cannot be combined with positional checkpoints."
        ),
    )
    parser.add_argument(
        "--log-root",
        default="logs/RobotDancing",
        help="Training log root used with --training-manifest.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip incomplete manifest entries instead of failing.",
    )
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument(
        "--rollouts",
        type=int,
        help="Number of one-second fixed starts; inferred from weights metadata.",
    )
    parser.add_argument("--start-interval-s", type=float, default=1.0)
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument(
        "--protocol",
        choices=("fixed_grid", "full_sequence", "both"),
        default="fixed_grid",
    )
    parser.add_argument(
        "--checkpoint-iterations",
        type=parse_iterations,
        default=(),
        help="Expand each supplied checkpoint/run directory to model_N.pt.",
    )
    parser.add_argument(
        "--full-sequence-rollouts",
        type=int,
        default=1,
        help="Repeated deterministic start-at-zero rollouts.",
    )
    parser.add_argument("--bin-size-s", type=float, default=15.0)
    parser.add_argument("--output-root", default="revision_results/evaluation")
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        help="Comma-separated physical GPU IDs; omitted means serial execution.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Execute entries even when rollouts.csv already exists.",
    )
    parser.add_argument(
        "--keep-motion-far-termination",
        action="store_true",
        help=(
            "Keep terminate_when_motion_far enabled during evaluation. "
            "By default revision evaluation disables motion-far termination "
            "so rollouts measure long-horizon drift instead of stopping at "
            "the tracking-distance threshold."
        ),
    )
    args = parser.parse_args()
    if bool(args.checkpoints) == bool(args.training_manifest):
        parser.error(
            "provide either positional checkpoints or --training-manifest"
        )

    protocols = (
        ("fixed_grid", "full_sequence")
        if args.protocol == "both"
        else (args.protocol,)
    )
    identities = {}
    if args.training_manifest:
        checkpoints, identities = expand_manifest_checkpoints(
            args.training_manifest,
            args.log_root,
            args.checkpoint_iterations,
            args.allow_missing,
        )
    else:
        checkpoints = expand_checkpoints(
            args.checkpoints, args.checkpoint_iterations
        )
    output_root = Path(args.output_root)
    records = []
    for checkpoint in checkpoints:
        for protocol in protocols:
            rollout_count = (
                args.full_sequence_rollouts
                if protocol == "full_sequence"
                else (
                    args.rollouts
                    if args.rollouts is not None
                    else infer_rollout_count(
                        checkpoint,
                        start_interval_s=args.start_interval_s,
                        start_offset_s=args.start_offset_s,
                    )
                )
            )
            if rollout_count <= 0:
                raise ValueError("rollouts must be positive")
            base_run_name = f"{checkpoint.parent.name}_{checkpoint.stem}"
            run_name = f"{base_run_name}_{protocol}"
            output_dir = output_root / "nominal" / protocol / base_run_name
            fixed_grid = protocol == "fixed_grid"
            command = [
                args.python_bin,
                "humanoidverse/eval_agent.py",
                f"+checkpoint={checkpoint}",
                "+domain_rand=NO_domain_rand",
                "+opt=eval_revision_metrics",
                f"eval_name={run_name}",
                f"num_envs={rollout_count}",
                f"env.config.eval_fixed_start_grid={str(fixed_grid)}",
                "env.config.enforce_randomize_motion_start_eval=False",
                (
                    "env.config.eval_fixed_start_interval_s="
                    f"{args.start_interval_s}"
                ),
                (
                    "env.config.eval_fixed_start_offset_s="
                    f"{args.start_offset_s}"
                ),
                (
                    "++env.config.termination.terminate_when_motion_far="
                    f"{str(args.keep_motion_far_termination)}"
                ),
                (
                    "++env.config.termination_curriculum."
                    "terminate_when_motion_far_curriculum="
                    f"{str(args.keep_motion_far_termination)}"
                ),
                (
                    "algo.config.eval_callbacks.revision_metrics.config."
                    f"rollout_count={rollout_count}"
                ),
                (
                    "algo.config.eval_callbacks.revision_metrics.config."
                    f"bin_size_s={args.bin_size_s}"
                ),
                (
                    "algo.config.eval_callbacks.revision_metrics.config."
                    f"output_dir={output_dir.resolve()}"
                ),
                (
                    "algo.config.eval_callbacks.revision_metrics.config."
                    f"start_interval_s={args.start_interval_s}"
                ),
                (
                    "algo.config.eval_callbacks.revision_metrics.config."
                    f"start_offset_s={args.start_offset_s}"
                ),
                (
                    "algo.config.eval_callbacks.revision_metrics.config."
                    f"protocol={protocol}"
                ),
            ]
            status = (
                "completed"
                if (output_dir / "rollouts.csv").is_file()
                else "pending"
            )
            records.append(
                {
                    "experiment_name": run_name,
                    "run_name": checkpoint.parent.name,
                    "checkpoint": str(checkpoint),
                    "iteration": checkpoint_iteration(checkpoint),
                    "protocol": protocol,
                    "output_dir": str(output_dir),
                    "rollout_count": rollout_count,
                    "bin_size_s": args.bin_size_s,
                    "start_interval_s": args.start_interval_s,
                    "start_offset_s": args.start_offset_s,
                    "motion_far_termination": (
                        "enabled"
                        if args.keep_motion_far_termination
                        else "disabled"
                    ),
                    "status": status,
                    "command": command,
                    **identities.get(str(checkpoint), {}),
                }
            )
            print(f"[{status}] {shlex.join(command)}", flush=True)

    manifest = output_root / "evaluation_manifest.json"
    write_manifest(manifest, records)
    print(f"Wrote manifest: {manifest}", flush=True)
    if args.run:
        run_experiments(
            records,
            gpu_ids=args.gpus,
            rerun_completed=args.rerun_completed,
        )
    write_progression_summary(output_root, records)


if __name__ == "__main__":
    main()
