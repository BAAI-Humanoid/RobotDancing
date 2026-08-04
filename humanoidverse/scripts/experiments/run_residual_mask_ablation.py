#!/usr/bin/env python3
"""Build E1/E2 residual-action and mask ablation matrices."""

from __future__ import annotations

import argparse
import hashlib
import shlex
from pathlib import Path

try:
    from .experiment_utils import (
        completed_run_dirs,
        load_robot_dof_names,
        parse_gpu_ids,
        resolve_joint_indices,
        run_experiments,
        validate_distribution_weights,
        write_manifest,
    )
    from .run_resampling_ablation import (
        DISTRIBUTION_SAMPLERS,
        KNOWN_CONDITIONS as KNOWN_SAMPLERS,
        MOTIONS,
        SAMPLER_FLAGS,
        parse_csv,
    )
except ImportError:
    from experiment_utils import (
        completed_run_dirs,
        load_robot_dof_names,
        parse_gpu_ids,
        resolve_joint_indices,
        run_experiments,
        validate_distribution_weights,
        write_manifest,
    )
    from run_resampling_ablation import (
        DISTRIBUTION_SAMPLERS,
        KNOWN_CONDITIONS as KNOWN_SAMPLERS,
        MOTIONS,
        SAMPLER_FLAGS,
        parse_csv,
    )


G1_CONFIG = "humanoidverse/config/robot/g1/g1_23dof_lock_wrist.yaml"
MASKS = {
    "selective": (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
    ),
    "hip_knee_ankle_pitch": (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
    ),
}
CONDITIONS = (
    "absolute_uniform",
    "absolute_final",
    "selective",
    "hip_knee_ankle_pitch",
    "all_legs",
    "legs_torso",
    "all_dofs",
)
SMOKE_CONDITIONS = (
    "absolute_final",
    "selective",
    "hip_knee_ankle_pitch",
    "all_dofs",
)
FINAL_MOTIONS = "dance1_subject1,dance1_subject2,dance2_subject3,dance2_subject4"
FINAL_SEEDS = "1,2,3"
FINAL_ITERATIONS = 50000
SMOKE_ITERATIONS = 5000


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampler_metadata(sampling, sampler_version):
    if sampling not in SAMPLER_FLAGS:
        raise ValueError(f"Unknown sampler: {sampling}")
    config_path = Path(f"humanoidverse/config/resampling/{sampling}.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return {
        "sampler": sampling,
        "sampler_version": sampler_version,
        "sampler_config_path": str(config_path),
        "sampler_config_sha256": sha256_file(config_path),
        **SAMPLER_FLAGS[sampling],
    }


def condition_settings(condition, final_sampling):
    dof_names = load_robot_dof_names(G1_CONFIG)
    if condition.startswith("absolute"):
        return False, [], "uniform" if condition == "absolute_uniform" else final_sampling
    if condition in MASKS:
        indices = resolve_joint_indices(G1_CONFIG, MASKS[condition])
    elif condition == "all_legs":
        indices = [
            index
            for index, name in enumerate(dof_names)
            if any(token in name for token in ("hip", "knee", "ankle"))
        ]
    elif condition == "legs_torso":
        indices = [
            index
            for index, name in enumerate(dof_names)
            if any(token in name for token in ("hip", "knee", "ankle", "waist"))
        ]
    elif condition == "all_dofs":
        indices = list(range(len(dof_names)))
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return True, indices, final_sampling


def build_records(args):
    motion_names = parse_csv(args.motions)
    unknown_motions = sorted(set(motion_names) - set(MOTIONS))
    if unknown_motions:
        raise ValueError(f"Unknown motions: {unknown_motions}")
    conditions = parse_csv(args.conditions)
    unknown_conditions = sorted(set(conditions) - set(CONDITIONS))
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}")
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.iterations <= 0 or args.num_envs <= 0:
        raise ValueError("iterations and num-envs must be positive")

    dof_names = load_robot_dof_names(G1_CONFIG)
    final_sampler = sampler_metadata(
        args.final_sampling,
        args.sampler_version,
    )
    records = []
    for motion_name in motion_names:
        motion_file, weights_file = MOTIONS[motion_name]
        if args.final_sampling in DISTRIBUTION_SAMPLERS:
            validate_distribution_weights(motion_file, weights_file)
        elif not Path(motion_file).is_file():
            raise FileNotFoundError(motion_file)
        for seed in seeds:
            for condition in conditions:
                enable_residual, indices, sampling = condition_settings(
                    condition, args.final_sampling
                )
                sampler = sampler_metadata(sampling, args.sampler_version)
                experiment_name = (
                    f"{motion_name}_{condition}_seed{seed}_"
                    f"{args.sampler_version}"
                )
                command = [
                    args.python_bin,
                    "humanoidverse/train_agent.py",
                    f"experiment_name={experiment_name}",
                    f"resampling={sampling}",
                    f"robot.motion.motion_file={motion_file}",
                    f"robot.motion.distribution_weights_path={weights_file}",
                    (
                        "robot.motion.enable_distribution_aware_resampling="
                        f"{sampler['enable_distribution_aware_resampling']}"
                    ),
                    (
                        "robot.motion.enable_failure_aware_resampling="
                        f"{sampler['enable_failure_aware_resampling']}"
                    ),
                    f"robot.control.enable_residual={str(enable_residual)}",
                    f"robot.control.residual_joint_indices={indices}",
                    f"seed={seed}",
                    f"num_envs={args.num_envs}",
                    f"algo.config.num_learning_iterations={args.iterations}",
                    "env.config.max_episode_length_s=200",
                    "rewards.set_reward=RD",
                    "rewards.locomotion_max_contact_force=400.0",
                    "rewards.reward_limit.soft_torque_limit=0.825",
                ]
                completed = completed_run_dirs(
                    args.log_root, experiment_name, args.iterations
                )
                records.append(
                    {
                        "experiment_name": experiment_name,
                        "study": (
                            "E1,E2"
                            if condition in ("selective", "all_dofs")
                            else (
                                "E1"
                                if condition in (
                                    "absolute_uniform",
                                    "absolute_final",
                                )
                                else "E2"
                            )
                        ),
                        "motion": motion_name,
                        "motion_file": motion_file,
                        "condition": condition,
                        "sampling": sampling,
                        "final_sampling": args.final_sampling,
                        "final_sampler_config_path": final_sampler[
                            "sampler_config_path"
                        ],
                        "final_sampler_config_sha256": final_sampler[
                            "sampler_config_sha256"
                        ],
                        "final_enable_distribution_aware_resampling": (
                            final_sampler[
                                "enable_distribution_aware_resampling"
                            ]
                        ),
                        "final_enable_failure_aware_resampling": (
                            final_sampler[
                                "enable_failure_aware_resampling"
                            ]
                        ),
                        "seed": seed,
                        "iterations": args.iterations,
                        "num_envs": args.num_envs,
                        "enable_residual": enable_residual,
                        "residual_joint_indices": indices,
                        "residual_joint_names": [
                            dof_names[index] for index in indices
                        ],
                        **sampler,
                        "protocol_id": (
                            f"residual_ablation_{args.sampler_version}_"
                            f"iter{args.iterations}_env{args.num_envs}"
                        ),
                        "command": command,
                        "status": "completed" if completed else "pending",
                        "completed_run_dirs": [
                            str(path) for path in completed
                        ],
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "final"),
        default="final",
        help=(
            "smoke: four motions, four diagnostic conditions, seed 1, "
            "5000 iterations; final: 84-run E1/E2 matrix."
        ),
    )
    parser.add_argument("--motions")
    parser.add_argument("--conditions")
    parser.add_argument("--seeds")
    parser.add_argument(
        "--final-sampling",
        choices=KNOWN_SAMPLERS,
        required=True,
    )
    parser.add_argument(
        "--sampler-version",
        required=True,
        help=(
            "Frozen sampler protocol label included in every experiment name. "
            "Change it whenever sampler semantics change."
        ),
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--log-root", default="logs/RobotDancing")
    parser.add_argument("--manifest", default="revision_manifests/residual_mask_ablation.json")
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        help="Comma-separated physical GPU IDs; omitted means serial execution.",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    if not args.sampler_version.strip():
        parser.error("--sampler-version must not be empty")
    if args.preset == "smoke":
        args.motions = args.motions or FINAL_MOTIONS
        args.conditions = args.conditions or ",".join(SMOKE_CONDITIONS)
        args.seeds = args.seeds or "1"
        args.iterations = args.iterations or SMOKE_ITERATIONS
    else:
        args.motions = args.motions or FINAL_MOTIONS
        args.conditions = args.conditions or ",".join(CONDITIONS)
        args.seeds = args.seeds or FINAL_SEEDS
        args.iterations = args.iterations or FINAL_ITERATIONS

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
