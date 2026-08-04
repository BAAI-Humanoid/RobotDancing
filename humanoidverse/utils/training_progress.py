"""Shared training progress, checkpoint, and scalar logging helpers."""

from __future__ import annotations

import re
from pathlib import Path


_CHECKPOINT_ITERATION_RE = re.compile(r"model_(\d+)\.pt$")


def completed_iteration(loop_index: int) -> int:
    """Convert a zero-based loop index to the number of completed iterations."""
    return int(loop_index) + 1


def restore_training_progress(
    checkpoint,
    num_steps_per_env: int,
    num_envs: int,
    checkpoint_path=None,
):
    """Restore progress while remaining compatible with legacy checkpoints."""
    iteration = int(checkpoint.get("iter", 0))

    # Legacy intermediate checkpoints were named with the current loop index
    # while storing a stale ``iter`` value. Use the filename only for those
    # checkpoints that do not contain the new explicit progress fields.
    if (
        checkpoint_path is not None
        and "tot_timesteps" not in checkpoint
        and "tot_time" not in checkpoint
    ):
        match = _CHECKPOINT_ITERATION_RE.search(Path(checkpoint_path).name)
        if match is not None:
            iteration = max(iteration, int(match.group(1)))

    total_timesteps = int(
        checkpoint.get(
            "tot_timesteps",
            iteration * int(num_steps_per_env) * int(num_envs),
        )
    )
    elapsed_seconds = float(checkpoint.get("tot_time", 0.0))
    return iteration, total_timesteps, elapsed_seconds


def training_progress_state(iteration, total_timesteps, elapsed_seconds):
    """Return progress fields stored in every new checkpoint."""
    return {
        "iter": int(iteration),
        "tot_timesteps": int(total_timesteps),
        "tot_time": float(elapsed_seconds),
    }


def _as_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def log_iteration_metrics(writer, metrics, iteration: int):
    """Write one iteration to TensorBoard and, if active, W&B exactly once."""
    iteration = int(iteration)
    scalar_metrics = {
        name: _as_scalar(value) for name, value in metrics.items()
    }
    scalar_metrics["Progress/iteration"] = iteration

    for name, value in scalar_metrics.items():
        writer.add_scalar(name, value, iteration)

    try:
        import wandb
    except ImportError:
        return scalar_metrics

    if wandb.run is not None:
        wandb.log(scalar_metrics, step=iteration)
    return scalar_metrics
