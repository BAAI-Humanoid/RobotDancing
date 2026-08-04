#!/usr/bin/env python3
"""Build H1/H1-2 quantitative cross-platform training matrices."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from experiment_utils import (
    completed_run_dirs,
    parse_gpu_ids,
    resolve_joint_indices,
    run_experiments,
    validate_distribution_weights,
    write_manifest,
)
from run_resampling_ablation import (
    DISTRIBUTION_SAMPLERS,
    SAMPLER_FLAGS,
    parse_csv,
)


PLATFORMS = {
    "h1": {
        "hydra_robot": "h1/h1_19dof",
        "config": "humanoidverse/config/robot/h1/h1_19dof.yaml",
        "motion_dir": "humanoidverse/data/motions/h1",
        "contact_force_limit": 650.0,
        "pitch_names": (
            "left_hip_pitch_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
        ),
    },
    "h1_2": {
        "hydra_robot": "h1_2/h1_2_27dof_anneal_21dof",
        "config": "humanoidverse/config/robot/h1_2/h1_2_27dof_anneal_21dof.yaml",
        "motion_dir": "humanoidverse/data/motions/h1_2",
        "contact_force_limit": 650.0,
        "pitch_names": (
            "left_hip_pitch_joint",
            "left_knee_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
        ),
    },
}
MOTIONS = (
    "dance1_subject1_0_3945",
    "dance1_subject2_0_3945",
    "dance2_subject3_0_6771",
    "dance2_subject4_0_6771",
)
def hydra_robot_override(platform):
    robot_group, robot_config = platform["hydra_robot"].rsplit("/", 1)
    return f"+robot/{robot_group}={robot_config}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platforms", default="h1,h1_2")
    parser.add_argument("--motions", default=",".join(MOTIONS))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--final-sampling",
        choices=(
            "uniform",
            "distribution_only",
            "failure_only",
            "combined",
            "combined_dominant",
        ),
        required=True,
    )
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--log-root", default="logs/RobotDancing")
    parser.add_argument("--manifest", default="revision_manifests/cross_platform.json")
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        help="Comma-separated physical GPU IDs; omitted means serial execution.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    records = []
    for platform_name in parse_csv(args.platforms):
        platform = PLATFORMS[platform_name]
        mask = resolve_joint_indices(
            platform["config"], platform["pitch_names"]
        )
        for motion_stem in parse_csv(args.motions):
            motion_file = f"{platform['motion_dir']}/{motion_stem}.pkl"
            weights_file = (
                f"{platform['motion_dir']}/{motion_stem}_"
                "distribution_segment_weights.json"
            )
            if args.final_sampling in DISTRIBUTION_SAMPLERS:
                validate_distribution_weights(motion_file, weights_file)
            elif not Path(motion_file).is_file():
                raise FileNotFoundError(motion_file)
            for seed in parse_csv(args.seeds, int):
                experiment_name = (
                    f"{platform_name}_{motion_stem}_{args.final_sampling}_seed{seed}"
                )
                sampler_flags = SAMPLER_FLAGS[args.final_sampling]
                command = [
                    args.python_bin,
                    "humanoidverse/train_agent.py",
                    hydra_robot_override(platform),
                    f"experiment_name={experiment_name}",
                    f"resampling={args.final_sampling}",
                    f"robot.motion.motion_file={motion_file}",
                    f"robot.control.residual_joint_indices={mask}",
                    f"seed={seed}",
                    f"num_envs={args.num_envs}",
                    f"algo.config.num_learning_iterations={args.iterations}",
                    (
                        "rewards.locomotion_max_contact_force="
                        f"{platform['contact_force_limit']}"
                    ),
                    (
                        "robot.motion.enable_distribution_aware_resampling="
                        f"{sampler_flags['enable_distribution_aware_resampling']}"
                    ),
                    (
                        "robot.motion.enable_failure_aware_resampling="
                        f"{sampler_flags['enable_failure_aware_resampling']}"
                    ),
                ]
                if args.final_sampling in DISTRIBUTION_SAMPLERS:
                    command.append(
                        f"robot.motion.distribution_weights_path={weights_file}"
                    )
                completed = completed_run_dirs(
                    args.log_root, experiment_name, args.iterations
                )
                records.append(
                    {
                        "experiment_name": experiment_name,
                        "study": "E4",
                        "platform": platform_name,
                        "motion": motion_stem,
                        "motion_file": motion_file,
                        "weights_file": (
                            weights_file
                            if args.final_sampling in DISTRIBUTION_SAMPLERS
                            else ""
                        ),
                        "seed": seed,
                        "condition": args.final_sampling,
                        "sampling": args.final_sampling,
                        "iterations": args.iterations,
                        "num_envs": args.num_envs,
                        "residual_joint_indices": mask,
                        **sampler_flags,
                        "locomotion_max_contact_force": (
                            platform["contact_force_limit"]
                        ),
                        "command": command,
                        "status": "completed" if completed else "pending",
                        "completed_run_dirs": [
                            str(path) for path in completed
                        ],
                    }
                )
    write_manifest(args.manifest, records)
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
