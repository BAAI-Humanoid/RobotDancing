#!/usr/bin/env python3
"""Build or execute ASAP and KungFuBot SOTA baseline matrices."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

try:
    from .experiment_utils import (
        completed_run_dirs,
        parse_gpu_ids,
        run_experiments,
        write_manifest,
    )
    from .run_resampling_ablation import MOTIONS, parse_csv
except ImportError:
    from experiment_utils import (
        completed_run_dirs,
        parse_gpu_ids,
        run_experiments,
        write_manifest,
    )
    from run_resampling_ablation import MOTIONS, parse_csv


BASELINES = {
    "asap": {
        "algorithm": "ppo",
        "obs_config": "asap",
        "reward_config": "reward_motion_tracking_asap",
        "use_vec_reward": False,
        "training_termination": {
            "terminate_when_motion_end": True,
            "terminate_when_motion_far": True,
            "terminate_by_orientation_error": False,
            "terminate_when_motion_far_curriculum": True,
            "terminate_when_motion_far_body_names": [],
            "terminate_when_motion_far_axes": [],
        },
        "command_overrides": (
            "algo=ppo",
            "env.config.use_vec_reward=False",
        ),
    },
    "kungfubot": {
        "algorithm": "mh_ppo",
        "obs_config": "kungfubot",
        "reward_config": "reward_motion_tracking_kungfubot",
        "use_vec_reward": True,
        "training_termination": {
            "terminate_when_motion_end": True,
            "terminate_when_motion_far": True,
            "terminate_by_orientation_error": False,
            "terminate_when_motion_far_curriculum": True,
            "terminate_when_motion_far_body_names": [],
            "terminate_when_motion_far_axes": [],
        },
        "command_overrides": (),
    },
}
FINAL_MOTIONS = "dance1_subject1,dance1_subject2,dance2_subject3,dance2_subject4"
FINAL_SEEDS = "1,2,3"
FINAL_ITERATIONS = 50000
SMOKE_MOTIONS = "dance1_subject2"
SMOKE_SEEDS = "1"
SMOKE_ITERATIONS = 5000


def hydra_list(values):
    if not values:
        return "[]"
    return "[" + ",".join(str(value) for value in values) + "]"


def termination_overrides(termination):
    body_names = hydra_list(
        termination["terminate_when_motion_far_body_names"]
    )
    axes = hydra_list(termination["terminate_when_motion_far_axes"])
    return [
        (
            "env.config.termination.terminate_when_motion_end="
            f"{termination['terminate_when_motion_end']}"
        ),
        (
            "env.config.termination.terminate_when_motion_far="
            f"{termination['terminate_when_motion_far']}"
        ),
        (
            "env.config.termination.terminate_by_orientation_error="
            f"{termination['terminate_by_orientation_error']}"
        ),
        (
            "env.config.termination.terminate_when_motion_far_body_names="
            f"{body_names}"
        ),
        (
            "env.config.termination.terminate_when_motion_far_axes="
            f"{axes}"
        ),
        (
            "env.config.termination_curriculum."
            "terminate_when_motion_far_curriculum="
            f"{termination['terminate_when_motion_far_curriculum']}"
        ),
    ]


def experiment_name_for(motion_name, baseline_name, seed):
    return f"{motion_name}_{baseline_name}_seed{seed}"


def build_command(
    args,
    baseline_name,
    baseline,
    motion_name,
    motion_file,
    seed,
):
    experiment_name = experiment_name_for(motion_name, baseline_name, seed)
    command = [
        args.python_bin,
        "humanoidverse/train_agent.py",
        f"obs/motion_tracking={baseline['obs_config']}",
        f"rewards/motion_tracking={baseline['reward_config']}",
        *baseline["command_overrides"],
        f"experiment_name={experiment_name}",
        "resampling=uniform",
        f"robot.motion.motion_file={motion_file}",
        "robot.control.enable_residual=False",
        "robot.control.residual_joint_indices=[]",
        "robot.motion.enable_failure_aware_resampling=False",
        "robot.motion.enable_distribution_aware_resampling=False",
        *termination_overrides(baseline["training_termination"]),
        f"seed={seed}",
        f"num_envs={args.num_envs}",
        f"algo.config.num_learning_iterations={args.iterations}",
    ]
    return command


def build_records(args):
    baseline_names = parse_csv(args.baselines)
    unknown_baselines = sorted(set(baseline_names) - set(BASELINES))
    if unknown_baselines:
        raise ValueError(f"Unknown baselines: {unknown_baselines}")
    motion_names = parse_csv(args.motions)
    unknown_motions = sorted(set(motion_names) - set(MOTIONS))
    if unknown_motions:
        raise ValueError(f"Unknown motions: {unknown_motions}")
    seeds = parse_csv(args.seeds, int)
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.iterations <= 0 or args.num_envs <= 0:
        raise ValueError("iterations and num-envs must be positive")

    records = []
    for motion_name in motion_names:
        motion_file, _weights_file = MOTIONS[motion_name]
        if not Path(motion_file).is_file():
            raise FileNotFoundError(motion_file)
        for seed in seeds:
            for baseline_name in baseline_names:
                baseline = BASELINES[baseline_name]
                experiment_name = experiment_name_for(
                    motion_name,
                    baseline_name,
                    seed,
                )
                command = build_command(
                    args,
                    baseline_name,
                    baseline,
                    motion_name,
                    motion_file,
                    seed,
                )
                completed = completed_run_dirs(
                    args.log_root,
                    experiment_name,
                    args.iterations,
                )
                records.append(
                    {
                        "experiment_name": experiment_name,
                        "study": "SOTA",
                        "baseline": baseline_name,
                        "algorithm": baseline["algorithm"],
                        "obs_config": baseline["obs_config"],
                        "reward_config": baseline["reward_config"],
                        "use_vec_reward": baseline["use_vec_reward"],
                        "motion": motion_name,
                        "motion_file": motion_file,
                        "condition": baseline_name,
                        "sampling": "uniform",
                        "seed": seed,
                        "iterations": args.iterations,
                        "num_envs": args.num_envs,
                        "enable_residual": False,
                        "residual_joint_indices": [],
                        "enable_failure_aware_resampling": False,
                        "enable_distribution_aware_resampling": False,
                        "training_termination": baseline[
                            "training_termination"
                        ],
                        "protocol_id": (
                            f"sota_baseline_iter{args.iterations}_"
                            f"env{args.num_envs}"
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
            "smoke: dance1_subject2, both baselines, seed 1, "
            "5000 iterations; final: four motions, both baselines, "
            "three seeds, 50000 iterations."
        ),
    )
    parser.add_argument("--baselines", default=",".join(BASELINES))
    parser.add_argument("--motions")
    parser.add_argument("--seeds")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--log-root", default="logs/RobotDancing")
    parser.add_argument(
        "--manifest",
        default="revision_manifests/sota_baseline_ablation.json",
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
        args.motions = args.motions or SMOKE_MOTIONS
        args.seeds = args.seeds or SMOKE_SEEDS
        args.iterations = args.iterations or SMOKE_ITERATIONS
    else:
        args.motions = args.motions or FINAL_MOTIONS
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
