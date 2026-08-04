#!/usr/bin/env python3
"""Build or execute the smoke and final resampling ablation matrices."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from experiment_utils import (
    completed_run_dirs,
    parse_gpu_ids,
    run_experiments,
    validate_distribution_weights,
    write_manifest,
)


CONDITIONS = (
    "uniform",
    "distribution_only",
    "failure_only",
    "combined",
)
KNOWN_CONDITIONS = CONDITIONS + (
    "combined_contextual",
    "combined_dominant",
    "combined_product",
)
DISTRIBUTION_SAMPLERS = (
    "distribution_only",
    "combined",
    "combined_contextual",
    "combined_dominant",
    "combined_product",
)
SAMPLER_FLAGS = {
    "uniform": {
        "enable_distribution_aware_resampling": False,
        "enable_failure_aware_resampling": False,
    },
    "distribution_only": {
        "enable_distribution_aware_resampling": True,
        "enable_failure_aware_resampling": False,
    },
    "failure_only": {
        "enable_distribution_aware_resampling": False,
        "enable_failure_aware_resampling": True,
    },
    "combined": {
        "enable_distribution_aware_resampling": True,
        "enable_failure_aware_resampling": True,
    },
    "combined_contextual": {
        "enable_distribution_aware_resampling": True,
        "enable_failure_aware_resampling": True,
    },
    "combined_dominant": {
        "enable_distribution_aware_resampling": True,
        "enable_failure_aware_resampling": True,
    },
    "combined_product": {
        "enable_distribution_aware_resampling": True,
        "enable_failure_aware_resampling": True,
    },
}
MOTIONS = {
    "dance1_subject1": (
        "humanoidverse/data/motions/g1/dance1_subject1_0_3945.pkl",
        "humanoidverse/data/motions/g1/dance1_subject1_0_3945_distribution_segment_weights.json",
    ),
    "dance1_subject2": (
        "humanoidverse/data/motions/g1/dance1_subject2_0_3945.pkl",
        "humanoidverse/data/motions/g1/dance1_subject2_0_3945_distribution_segment_weights.json",
    ),
    "dance2_subject3": (
        "humanoidverse/data/motions/g1/dance2_subject3_0_6771.pkl",
        "humanoidverse/data/motions/g1/dance2_subject3_0_6771_distribution_segment_weights.json",
    ),
    "dance2_subject4": (
        "humanoidverse/data/motions/g1/dance2_subject4_0_6771.pkl",
        "humanoidverse/data/motions/g1/dance2_4_distribution_segment_weights.json",
    ),
}
FINAL_MOTIONS = "dance1_subject1,dance1_subject2,dance2_subject3,dance2_subject4"


def parse_csv(value, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def experiment_name_for(args, condition, motion_name, seed):
    name = f"{motion_name}_{condition}_seed{seed}"
    if args.sampler_version:
        name = f"{name}_{args.sampler_version}"
    return name


def build_command(args, condition, motion_name, motion_file, weights_file, seed):
    experiment_name = experiment_name_for(
        args, condition, motion_name, seed
    )
    return [
        args.python_bin,
        "humanoidverse/train_agent.py",
        f"experiment_name={experiment_name}",
        f"resampling={condition}",
        f"robot.motion.motion_file={motion_file}",
        f"robot.motion.distribution_weights_path={weights_file}",
        f"seed={seed}",
        f"num_envs={args.num_envs}",
        f"algo.config.num_learning_iterations={args.iterations}",
        "env.config.max_episode_length_s=200",
        "rewards.set_reward=RD",
        "rewards.locomotion_max_contact_force=400.0",
        "rewards.reward_limit.soft_torque_limit=0.825",
    ]


def build_records(args):
    conditions = parse_csv(args.conditions)
    unknown_conditions = sorted(set(conditions) - set(KNOWN_CONDITIONS))
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    motion_names = parse_csv(args.motions)
    unknown_motions = sorted(set(motion_names) - set(MOTIONS))
    if unknown_motions:
        raise ValueError(f"Unknown motions: {unknown_motions}")
    seeds = parse_csv(args.seeds, int)

    records = []
    for motion_name in motion_names:
        motion_file, weights_file = MOTIONS[motion_name]
        validate_distribution_weights(motion_file, weights_file)
        for seed in seeds:
            for condition in conditions:
                experiment_name = experiment_name_for(
                    args, condition, motion_name, seed
                )
                command = build_command(
                    args,
                    condition,
                    motion_name,
                    motion_file,
                    weights_file,
                    seed,
                )
                completed = completed_run_dirs(
                    args.log_root, experiment_name, args.iterations
                )
                records.append(
                    {
                        "experiment_name": experiment_name,
                        "motion": motion_name,
                        "motion_file": motion_file,
                        "weights_file": weights_file,
                        "condition": condition,
                        "seed": seed,
                        "iterations": args.iterations,
                        "num_envs": args.num_envs,
                        "command": command,
                        "status": "completed" if completed else "pending",
                        "completed_run_dirs": [str(path) for path in completed],
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "final"),
        default="final",
        help="smoke: dance1_subject2/seed1/5000; final: four motions/three seeds/50000.",
    )
    parser.add_argument("--motions")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument(
        "--sampler-version",
        default="frontier_v5",
        help="Suffix for non-uniform runs so changed samplers are not skipped.",
    )
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--log-root", default="logs/RobotDancing")
    parser.add_argument(
        "--manifest",
        default="revision_manifests/resampling_ablation.json",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        help="Comma-separated physical GPU IDs; omitted means serial execution.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Execute entries even when their final checkpoint already exists.",
    )
    args = parser.parse_args()
    if args.preset == "smoke":
        args.motions = args.motions or "dance1_subject2"
        args.seeds = args.seeds or "1"
        args.iterations = args.iterations or 5000
    else:
        args.motions = args.motions or FINAL_MOTIONS
        args.seeds = args.seeds or "1,2,3"
        args.iterations = args.iterations or 50000

    records = build_records(args)
    write_manifest(args.manifest, records)
    print(f"Wrote manifest: {Path(args.manifest)}")
    for record in records:
        print(f"[{record['status']}] {shlex.join(record['command'])}", flush=True)
    if args.run:
        run_experiments(
            records,
            gpu_ids=args.gpus,
            rerun_completed=args.rerun_completed,
        )


if __name__ == "__main__":
    main()
