import unittest
from types import SimpleNamespace
from unittest.mock import patch

from humanoidverse.utils.training_progress import (
    completed_iteration,
    log_iteration_metrics,
    restore_training_progress,
    training_progress_state,
)


class FakeWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, name, value, step):
        self.calls.append((name, value, step))


class TrainingProgressTest(unittest.TestCase):
    def test_iteration_axis_does_not_depend_on_environment_count(self):
        for num_envs in (4096, 8192):
            writer = FakeWriter()
            iteration = completed_iteration(9999)
            total_timesteps = iteration * 24 * num_envs
            log_iteration_metrics(
                writer,
                {"Progress/total_timesteps": total_timesteps},
                iteration,
            )
            self.assertEqual(iteration, 10000)
            self.assertTrue(
                all(step == 10000 for _, _, step in writer.calls)
            )
            self.assertIn(
                ("Progress/total_timesteps", total_timesteps, 10000),
                writer.calls,
            )

    def test_new_checkpoint_progress_round_trip(self):
        checkpoint = training_progress_state(500, 500 * 24 * 4096, 12.5)
        progress = restore_training_progress(checkpoint, 24, 4096)
        self.assertEqual(progress, (500, 500 * 24 * 4096, 12.5))

    def test_legacy_checkpoint_derives_timestep_count(self):
        progress = restore_training_progress({"iter": 500}, 24, 4096)
        self.assertEqual(progress, (500, 500 * 24 * 4096, 0.0))

    def test_legacy_intermediate_checkpoint_uses_filename(self):
        progress = restore_training_progress(
            {"iter": 0},
            24,
            4096,
            checkpoint_path="/tmp/model_10000.pt",
        )
        self.assertEqual(progress, (10000, 10000 * 24 * 4096, 0.0))

    def test_wandb_receives_one_explicit_iteration_step(self):
        calls = []
        fake_wandb = SimpleNamespace(
            run=object(),
            log=lambda metrics, step: calls.append((metrics, step)),
        )
        writer = FakeWriter()
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            log_iteration_metrics(writer, {"Loss/value": 1.5}, 12)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 12)
        self.assertEqual(calls[0][0]["Progress/iteration"], 12)


if __name__ == "__main__":
    unittest.main()
