import torch
import numpy as np
from pathlib import Path
import json
import os
from typing import Dict, Optional
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    quat_conjugate,
    quat_to_angle_axis,
    get_euler_xyz_in_tensor,
)
from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.skeleton import SkeletonTree

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.envs.env_utils.resampling_utils import (
    accumulate_episode_statistics,
    accumulate_predecessor_failure_credit,
    apply_distribution_floor,
    classify_frontier_targets,
    combined_mixture_weights,
    compute_hard_window_scores,
    compute_sampling_probabilities,
    compute_segment_layout,
    contextual_sampling_pool_masks,
    dominant_contextual_sampling_pool_masks,
    distribution_failure_topk_conflict,
    eligible_episode_mask,
    failure_ramp,
    frontier_crossing_success,
    hard_window_start_times,
    mask_segments_by_max_time,
    masked_uniform,
    normalize_sparse_hard_window_probabilities,
    prior_alignment_diagnostics,
    sampling_summary,
    select_earliest_frontier_windows,
    select_hard_windows_nms,
    segment_time_bounds,
    should_update_failure_statistics,
    shrink_failure_rates,
    suppress_immediate_failure_probabilities,
    update_failure_ema,
    update_hard_window_lookbacks,
)
from humanoidverse.envs.env_utils.motion_far_termination import (
    compute_motion_far_errors,
    compute_quat_angle_error,
    compute_root_orientation_error,
    resolve_motion_far_termination_config,
)

from termcolor import colored
from loguru import logger

from scipy.spatial.transform import Rotation as sRot
import joblib

class LeggedRobotMotionTracking(LeggedRobotBase):
    def __init__(self, config, device):
        self.init_done = False
        self.debug_viz = True
        
        super().__init__(config, device)
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_motion_far_termination_config()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True

        self._init_save_motion()

        if self.config.use_teleop_control:
            self.teleop_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import Float64MultiArray
            self.node = Node("motion_tracking")
            self.teleop_sub = self.node.create_subscription(Float64MultiArray, "vision_pro_data", self.teleop_callback, 1)

        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum:
            self.terminate_when_motion_far_threshold = self.config.termination_curriculum.terminate_when_motion_far_initial_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        else:
            self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
            logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

        if self.config.termination.terminate_when_dof_far:
            self.terminate_when_dof_far_threshold = self.config.termination_curriculum.terminate_when_dof_far_initial_threshold
            logger.info(f"Terminate when dof far threshold: {self.terminate_when_dof_far_threshold}")
        
        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable:
            self.sdc_alpha = self.config.soft_dynamic_correction.alpha
            logger.info(f"Soft Dynamic Correction alpha: {self.sdc_alpha}")

    def teleop_callback(self, msg):
        self.teleop_marker_coords = torch.tensor(msg.data, device=self.device)

    def _init_save_motion(self):
        if "save_motion" in self.config:
            self.save_motion = self.config.save_motion
            if self.save_motion:
                os.makedirs(Path(self.config.ckpt_dir) / "motions", exist_ok = True)
                
                if hasattr(self.config, 'dump_motion_name'):
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / (str(self.config.eval_timestamp) + "_" + self.config.dump_motion_name)
                else:
                    self.save_motion_dir = Path(self.config.ckpt_dir) / "motions" / f"{self.config.save_note}_{self.config.eval_timestamp}"
                self.save_motion = True
                self.num_augment_joint = len(self.config.robot.motion.extend_config)
                self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                            'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[], 'contact_mask':[]}
                self.motion_times_buf = []
                self.start_save = False

        else:
            self.save_motion = False

    def _init_motion_lib(self):
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        memory_optimize = getattr(self.config.robot.motion, "memory_optimize", True)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False, memory_optimize=memory_optimize)
        else:
            self._motion_lib.load_motions(random_sample=True, memory_optimize=memory_optimize)
            
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        self.num_motions = self._motion_lib._num_unique_motions

        self._init_segment_motion_resampling()

        ref_init_state = self.kick_motion_res()
        self._kick_motion_res_counter = -1
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        res = self._resample_motion_times(torch.arange(self.num_envs))

        if hasattr(self._motion_lib, "has_contact_mask") and self._motion_lib.has_contact_mask:
            self.ref_contact_mask = torch.zeros(self.num_envs, self._motion_lib._contact_size, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_segment_motion_resampling(self):
        """Initialize segment-based motion-time resampling."""
        self.enable_failure_aware_resampling = getattr(
            self.config.robot.motion, 'enable_failure_aware_resampling', False
        )
        self.enable_distribution_aware_resampling = getattr(
            self.config.robot.motion, 'enable_distribution_aware_resampling', False
        )
        self.enable_segment_resampling = (
            self.enable_failure_aware_resampling or self.enable_distribution_aware_resampling
        )
        if not self.enable_segment_resampling:
            logger.info("Segment motion resampling disabled")
            return

        motion_config = self.config.robot.motion
        self.segment_duration_s = float(getattr(motion_config, "segment_duration_s", 1.0))
        self.failure_rate_ema_alpha = float(getattr(motion_config, "failure_rate_ema_alpha", 0.05))
        self.failure_bias_strength = float(getattr(motion_config, "failure_bias_strength", 2.0))
        self.distribution_strength = float(getattr(motion_config, "distribution_strength", 0.5))
        self.combined_mode = str(getattr(motion_config, "combined_mode", "mixture"))
        self.combined_failure_weight = float(
            getattr(motion_config, "combined_failure_weight", 0.4)
        )
        self.combined_distribution_weight = float(
            getattr(motion_config, "combined_distribution_weight", 0.2)
        )
        self.combined_distribution_weight_final = float(
            getattr(
                motion_config,
                "combined_distribution_weight_final",
                self.combined_distribution_weight,
            )
        )
        self.combined_failure_weight_final = float(
            getattr(
                motion_config,
                "combined_failure_weight_final",
                self.combined_failure_weight,
            )
        )
        self.combined_distribution_guard = bool(
            getattr(motion_config, "combined_distribution_guard", False)
        )
        self.combined_guard_refresh_steps = int(
            getattr(motion_config, "combined_guard_refresh_steps", 12000)
        )
        self.combined_guard_topk_fraction = float(
            getattr(motion_config, "combined_guard_topk_fraction", 0.1)
        )
        self.combined_guard_disable_steps = int(
            getattr(motion_config, "combined_guard_disable_steps", 24000)
        )
        self.combined_guard_conflict_threshold = float(
            getattr(motion_config, "combined_guard_conflict_threshold", 0.75)
        )
        self.combined_guard_gate_loss_threshold = float(
            getattr(motion_config, "combined_guard_gate_loss_threshold", 0.25)
        )
        self.combined_guard_min_failure_confidence = float(
            getattr(motion_config, "combined_guard_min_failure_confidence", 0.05)
        )
        self.combined_guard_pool_survival_margin_s = float(
            getattr(motion_config, "combined_guard_pool_survival_margin_s", 2.0)
        )
        self.combined_force_failure_disable_steps = int(
            getattr(
                motion_config,
                "combined_force_failure_disable_steps",
                self.combined_guard_disable_steps,
            )
        )
        self.tracking_aware_hard_window = bool(
            getattr(motion_config, "tracking_aware_hard_window", False)
        )
        self.tracking_error_ema_alpha = float(
            getattr(motion_config, "tracking_error_ema_alpha", 0.05)
        )
        self.tracking_error_scale = float(
            getattr(motion_config, "tracking_error_scale", 0.35)
        )
        self.tracking_quality_floor = float(
            getattr(motion_config, "tracking_quality_floor", 0.2)
        )
        self.tracking_joint_error_weight = float(
            getattr(motion_config, "tracking_joint_error_weight", 0.25)
        )
        self.failure_priority_mode = str(
            getattr(motion_config, "failure_priority_mode", "frontier")
        )
        self.hard_window_replay = bool(
            getattr(motion_config, "hard_window_replay", False)
        )
        self.hard_window_ratio = float(
            getattr(motion_config, "hard_window_ratio", 0.2)
        )
        self.hard_window_distribution_ratio = float(
            getattr(motion_config, "hard_window_distribution_ratio", 0.2)
        )
        self.hard_window_failure_ratio = float(
            getattr(motion_config, "hard_window_failure_ratio", 0.0)
        )
        self.hard_window_use_combined_distribution_weight = bool(
            getattr(
                motion_config,
                "hard_window_use_combined_distribution_weight",
                False,
            )
        )
        self.hard_window_top_k = int(
            getattr(motion_config, "hard_window_top_k", 8)
        )
        self.hard_window_lookback_s = float(
            getattr(motion_config, "hard_window_lookback_s", 4.0)
        )
        self.hard_window_jitter_s = float(
            getattr(motion_config, "hard_window_jitter_s", 0.5)
        )
        self.hard_window_refresh_steps = int(
            getattr(motion_config, "hard_window_refresh_steps", 24000)
        )
        self.hard_window_max_probability_ratio = float(
            getattr(
                motion_config,
                "hard_window_max_probability_ratio",
                8.0,
            )
        )
        self.failure_credit_lookback_s = float(
            getattr(motion_config, "failure_credit_lookback_s", 5.0)
        )
        self.failure_credit_decay_s = float(
            getattr(motion_config, "failure_credit_decay_s", 2.0)
        )
        self.hard_window_immediate_failure_s = float(
            getattr(
                motion_config,
                "hard_window_immediate_failure_s",
                1.0,
            )
        )
        self.hard_window_immediate_failure_threshold = float(
            getattr(
                motion_config,
                "hard_window_immediate_failure_threshold",
                0.35,
            )
        )
        self.hard_window_immediate_failure_min_exposures = int(
            getattr(
                motion_config,
                "hard_window_immediate_failure_min_exposures",
                32,
            )
        )
        self.hard_window_terminal_suppression_floor = float(
            getattr(
                motion_config,
                "hard_window_terminal_suppression_floor",
                0.1,
            )
        )
        self.suppress_immediate_failure_terminal_probability = bool(
            getattr(
                motion_config,
                "suppress_immediate_failure_terminal_probability",
                False,
            )
        )
        self.hard_window_max_lookback_s = float(
            getattr(motion_config, "hard_window_max_lookback_s", 8.0)
        )
        self.failure_regression_weight = float(
            getattr(motion_config, "failure_regression_weight", 0.0)
        )
        self.frontier_window_ratio = float(
            getattr(motion_config, "frontier_window_ratio", 0.2)
        )
        self.frontier_window_top_k = int(
            getattr(motion_config, "frontier_window_top_k", 4)
        )
        self.frontier_window_refresh_steps = int(
            getattr(motion_config, "frontier_window_refresh_steps", 24000)
        )
        self.frontier_window_max_probability_ratio = float(
            getattr(
                motion_config,
                "frontier_window_max_probability_ratio",
                16.0,
            )
        )
        self.frontier_failure_threshold = float(
            getattr(motion_config, "frontier_failure_threshold", 0.35)
        )
        self.frontier_confidence_threshold = float(
            getattr(motion_config, "frontier_confidence_threshold", 0.1)
        )
        self.frontier_span_s = float(
            getattr(motion_config, "frontier_span_s", 20.0)
        )
        self.frontier_earliest_contiguous = bool(
            getattr(motion_config, "frontier_earliest_contiguous", False)
        )
        self.frontier_crossing_margin_s = float(
            getattr(
                motion_config,
                "frontier_crossing_margin_s",
                self.segment_duration_s,
            )
        )
        self.frontier_advance_success_threshold = float(
            getattr(
                motion_config,
                "frontier_advance_success_threshold",
                0.6,
            )
        )
        self.frontier_min_attempts = int(
            getattr(motion_config, "frontier_min_attempts", 256)
        )
        self.frontier_mastered_threshold = float(
            getattr(motion_config, "frontier_mastered_threshold", 0.8)
        )
        self.frontier_impossible_threshold = float(
            getattr(motion_config, "frontier_impossible_threshold", 0.02)
        )
        self.min_samples_for_stats = int(getattr(motion_config, "min_samples_for_stats", 1000))
        self.failure_confidence_exposure = int(
            getattr(motion_config, "failure_confidence_exposure", 10000)
        )
        self.failure_stats_update_interval_steps = int(
            getattr(motion_config, "failure_stats_update_interval_steps", 100)
        )
        self.failure_rate_epsilon = float(getattr(motion_config, "failure_rate_epsilon", 1e-3))
        self.distribution_min_uniform_ratio = float(
            getattr(motion_config, "distribution_min_uniform_ratio", 0.1)
        )
        self.uniform_mix_ratio = float(getattr(motion_config, "uniform_mix_ratio", 0.3))
        self.max_sampling_probability_ratio = float(
            getattr(motion_config, "max_sampling_probability_ratio", 4.0)
        )
        self.distribution_warmup_steps = int(
            getattr(motion_config, "distribution_warmup_steps", 24000)
        )
        self.distribution_ramp_steps = int(
            getattr(motion_config, "distribution_ramp_steps", 24000)
        )
        self.failure_warmup_steps = int(getattr(motion_config, "failure_warmup_steps", 48000))
        self.failure_ramp_steps = int(getattr(motion_config, "failure_ramp_steps", 48000))
        self.combined_handoff_start_steps = int(
            getattr(
                motion_config,
                "combined_handoff_start_steps",
                self.failure_warmup_steps,
            )
        )
        self.combined_handoff_ramp_steps = int(
            getattr(
                motion_config,
                "combined_handoff_ramp_steps",
                self.failure_ramp_steps,
            )
        )
        self.strict_distribution_weights = bool(
            getattr(motion_config, "strict_distribution_weights", True)
        )
        self.resampling_log_interval_steps = int(
            getattr(motion_config, "resampling_log_interval_steps", 100)
        )
        self.resampling_snapshot_interval_steps = int(
            getattr(motion_config, "resampling_snapshot_interval_steps", 12000)
        )
        if self.resampling_log_interval_steps <= 0:
            raise ValueError("resampling_log_interval_steps must be positive")
        if self.resampling_snapshot_interval_steps <= 0:
            raise ValueError("resampling_snapshot_interval_steps must be positive")
        if self.failure_stats_update_interval_steps <= 0:
            raise ValueError(
                "failure_stats_update_interval_steps must be positive"
            )
        if self.failure_confidence_exposure <= 0:
            raise ValueError("failure_confidence_exposure must be positive")
        if self.segment_duration_s <= 0:
            raise ValueError("segment_duration_s must be positive")
        if self.min_samples_for_stats < 0:
            raise ValueError("min_samples_for_stats must be non-negative")
        if self.distribution_strength <= 0:
            raise ValueError("distribution_strength must be positive")
        if self.combined_mode not in ("mixture", "product"):
            raise ValueError("combined_mode must be 'mixture' or 'product'")
        if not 0.0 <= self.combined_failure_weight <= 1.0:
            raise ValueError("combined_failure_weight must be in [0, 1]")
        if not 0.0 <= self.combined_distribution_weight <= 1.0:
            raise ValueError("combined_distribution_weight must be in [0, 1]")
        if (
            self.combined_distribution_weight
            + self.combined_failure_weight
            > 1.0
        ):
            raise ValueError(
                "combined distribution and failure weights must sum to at most 1"
            )
        if not 0.0 <= self.combined_failure_weight_final <= 1.0:
            raise ValueError(
                "combined_failure_weight_final must be in [0, 1]"
            )
        if not 0.0 <= self.combined_distribution_weight_final <= 1.0:
            raise ValueError(
                "combined_distribution_weight_final must be in [0, 1]"
            )
        if (
            self.combined_distribution_weight_final
            + self.combined_failure_weight_final
            > 1.0
        ):
            raise ValueError(
                "final combined distribution and failure weights must sum "
                "to at most 1"
            )
        if self.combined_handoff_start_steps < 0:
            raise ValueError(
                "combined_handoff_start_steps must be non-negative"
            )
        if self.combined_handoff_ramp_steps < 0:
            raise ValueError(
                "combined_handoff_ramp_steps must be non-negative"
            )
        if self.combined_guard_refresh_steps <= 0:
            raise ValueError("combined_guard_refresh_steps must be positive")
        if self.combined_guard_disable_steps < 0:
            raise ValueError(
                "combined_guard_disable_steps must be non-negative"
            )
        if self.combined_force_failure_disable_steps < 0:
            raise ValueError(
                "combined_force_failure_disable_steps must be non-negative"
            )
        if not 0.0 < self.combined_guard_topk_fraction <= 1.0:
            raise ValueError(
                "combined_guard_topk_fraction must be in (0, 1]"
            )
        if not 0.0 <= self.combined_guard_conflict_threshold <= 1.0:
            raise ValueError(
                "combined_guard_conflict_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.combined_guard_gate_loss_threshold <= 1.0:
            raise ValueError(
                "combined_guard_gate_loss_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.combined_guard_min_failure_confidence <= 1.0:
            raise ValueError(
                "combined_guard_min_failure_confidence must be in [0, 1]"
            )
        if self.combined_guard_pool_survival_margin_s < 0.0:
            raise ValueError(
                "combined_guard_pool_survival_margin_s must be non-negative"
            )
        if not 0.0 <= self.tracking_error_ema_alpha <= 1.0:
            raise ValueError("tracking_error_ema_alpha must be in [0, 1]")
        if self.tracking_error_scale <= 0.0:
            raise ValueError("tracking_error_scale must be positive")
        if not 0.0 <= self.tracking_quality_floor <= 1.0:
            raise ValueError("tracking_quality_floor must be in [0, 1]")
        if self.tracking_joint_error_weight < 0.0:
            raise ValueError(
                "tracking_joint_error_weight must be non-negative"
            )
        if self.failure_priority_mode not in ("raw", "frontier"):
            raise ValueError(
                "failure_priority_mode must be 'raw' or 'frontier'"
            )
        if self.hard_window_replay and not (
            self.enable_distribution_aware_resampling
            and self.enable_failure_aware_resampling
        ):
            raise ValueError(
                "hard_window_replay requires combined distribution- and "
                "failure-aware resampling"
            )
        if not 0.0 <= self.hard_window_ratio <= 1.0:
            raise ValueError("hard_window_ratio must be in [0, 1]")
        if not 0.0 <= self.hard_window_distribution_ratio <= 1.0:
            raise ValueError(
                "hard_window_distribution_ratio must be in [0, 1]"
            )
        if not 0.0 <= self.hard_window_failure_ratio <= 1.0:
            raise ValueError("hard_window_failure_ratio must be in [0, 1]")
        if (
            self.hard_window_ratio
            + self.frontier_window_ratio
            + self.hard_window_distribution_ratio
            + self.hard_window_failure_ratio
            > 1.0
        ):
            raise ValueError(
                "frontier, hard-window, distribution, and failure pool "
                "ratios must sum to at most 1"
            )
        if self.hard_window_top_k <= 0:
            raise ValueError("hard_window_top_k must be positive")
        if self.hard_window_lookback_s < 0:
            raise ValueError("hard_window_lookback_s must be non-negative")
        if self.hard_window_jitter_s < 0:
            raise ValueError("hard_window_jitter_s must be non-negative")
        if self.hard_window_refresh_steps <= 0:
            raise ValueError("hard_window_refresh_steps must be positive")
        if self.hard_window_max_probability_ratio < 1.0:
            raise ValueError(
                "hard_window_max_probability_ratio must be at least 1"
            )
        if self.failure_credit_decay_s <= 0:
            raise ValueError("failure_credit_decay_s must be positive")
        if self.failure_credit_lookback_s < 0:
            raise ValueError("failure_credit_lookback_s must be non-negative")
        if self.hard_window_max_lookback_s < self.hard_window_lookback_s:
            raise ValueError(
                "hard_window_max_lookback_s must be at least the default "
                "lookback"
            )
        if not 0.0 <= self.hard_window_immediate_failure_threshold < 1.0:
            raise ValueError(
                "hard_window_immediate_failure_threshold must be in [0, 1)"
            )
        if self.hard_window_immediate_failure_min_exposures < 0:
            raise ValueError(
                "hard_window_immediate_failure_min_exposures must be "
                "non-negative"
            )
        if not 0.0 <= self.hard_window_terminal_suppression_floor <= 1.0:
            raise ValueError(
                "hard_window_terminal_suppression_floor must be in [0, 1]"
            )
        if self.failure_regression_weight < 0.0:
            raise ValueError("failure_regression_weight must be non-negative")
        if not 0.0 <= self.frontier_window_ratio <= 1.0:
            raise ValueError("frontier_window_ratio must be in [0, 1]")
        if self.frontier_window_top_k <= 0:
            raise ValueError("frontier_window_top_k must be positive")
        if self.frontier_window_refresh_steps <= 0:
            raise ValueError(
                "frontier_window_refresh_steps must be positive"
            )
        if self.frontier_window_max_probability_ratio < 1.0:
            raise ValueError(
                "frontier_window_max_probability_ratio must be at least 1"
            )
        if not 0.0 <= self.frontier_failure_threshold <= 1.0:
            raise ValueError(
                "frontier_failure_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.frontier_confidence_threshold <= 1.0:
            raise ValueError(
                "frontier_confidence_threshold must be in [0, 1]"
            )
        if self.frontier_span_s <= 0:
            raise ValueError("frontier_span_s must be positive")
        if self.frontier_crossing_margin_s <= 0:
            raise ValueError("frontier_crossing_margin_s must be positive")
        if not 0.0 <= self.frontier_advance_success_threshold <= 1.0:
            raise ValueError(
                "frontier_advance_success_threshold must be in [0, 1]"
            )
        if self.frontier_min_attempts <= 0:
            raise ValueError("frontier_min_attempts must be positive")
        if not (
            0.0
            <= self.frontier_impossible_threshold
            < self.frontier_mastered_threshold
            <= 1.0
        ):
            raise ValueError(
                "frontier thresholds must satisfy "
                "0 <= impossible < mastered <= 1"
            )

        self.unique_motion_lengths, self.unique_motion_frame_dts = (
            self._motion_lib.get_unique_motion_metadata()
        )
        self.motion_segment_counts, self.motion_segment_valid_mask = compute_segment_layout(
            self.unique_motion_lengths, self.segment_duration_s
        )
        self.max_num_time_segments = self.motion_segment_valid_mask.shape[1]
        if (
            self.hard_window_replay
            and self.hard_window_top_k
            * self.hard_window_max_probability_ratio
            < self.max_num_time_segments * self.hard_window_ratio
        ):
            raise ValueError(
                "hard_window_top_k is too small to satisfy the requested "
                "hard-window probability cap"
            )
        if (
            self.hard_window_replay
            and self.frontier_window_top_k
            * self.frontier_window_max_probability_ratio
            < self.max_num_time_segments * self.frontier_window_ratio
        ):
            raise ValueError(
                "frontier_window_top_k is too small to satisfy the "
                "requested frontier probability cap"
            )
        # Backward-compatible name for downstream logging and diagnostics.
        self.num_time_segments = self.max_num_time_segments

        uniform = masked_uniform(self.motion_segment_valid_mask)
        self.distribution_segment_weights = uniform.clone()
        if self.enable_distribution_aware_resampling:
            self._load_distribution_resampling_weights()

        tensor_shape = (self.num_motions, self.max_num_time_segments)
        self.motion_segment_failure_rates = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.motion_segment_start_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.motion_segment_exposure_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.motion_segment_failure_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self._pending_segment_exposure_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self._pending_segment_failure_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.motion_segment_predecessor_credit = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self._pending_segment_predecessor_credit = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.motion_segment_learning_progress = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.motion_segment_failure_regression = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self._previous_failure_rates = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.motion_segment_tracking_error_ema = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.motion_segment_tracking_error_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self._pending_segment_tracking_error_sums = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self._pending_segment_tracking_error_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.hard_window_immediate_exposures = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.hard_window_target_counts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.hard_window_immediate_failures = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.hard_window_immediate_failure_rates = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.hard_window_terminal_probability_multiplier = torch.ones(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.hard_window_dynamic_lookback_s = torch.full(
            tensor_shape,
            self.hard_window_lookback_s,
            dtype=torch.float32,
            device=self.device,
        )
        self.hard_window_scores = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self.hard_window_mask = torch.zeros(
            tensor_shape, dtype=torch.bool, device=self.device
        )
        self.hard_window_probabilities = uniform.clone()
        self.frontier_window_mask = torch.zeros(
            tensor_shape, dtype=torch.bool, device=self.device
        )
        self.frontier_window_probabilities = uniform.clone()
        self.frontier_anchor_segments = torch.full(
            (self.num_motions,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.frontier_window_attempts = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self.frontier_window_successes = torch.zeros(
            tensor_shape, dtype=torch.long, device=self.device
        )
        self._last_frontier_success_rate = torch.zeros(
            self.num_motions, dtype=torch.float32, device=self.device
        )
        self._frontier_mastered_mask = torch.zeros(
            tensor_shape, dtype=torch.bool, device=self.device
        )
        self._frontier_impossible_mask = torch.zeros(
            tensor_shape, dtype=torch.bool, device=self.device
        )
        self.env_current_segments = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_motion_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_hard_window_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.env_frontier_window_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.env_resampling_pool_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_hard_target_segments = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._resampling_pool_survival_ema = torch.zeros(
            5, dtype=torch.float32, device=self.device
        )
        self._resampling_pool_success_ema = torch.zeros(
            5, dtype=torch.float32, device=self.device
        )
        self._resampling_pool_counts = torch.zeros(
            5, dtype=torch.long, device=self.device
        )
        self._combined_guard_disabled_until_step = -1
        self._combined_force_failure_until_step = -1
        self._combined_guard_trigger_count = 0
        self._combined_force_failure_count = 0
        self._last_combined_guard_step = -self.combined_guard_refresh_steps
        self._last_distribution_conflict = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._last_combined_guard_gate_loss = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._window_segment_start_counts = torch.zeros_like(
            self.motion_segment_start_counts
        )
        self._window_expected_probability_mass = torch.zeros(
            tensor_shape, dtype=torch.float32, device=self.device
        )
        self._window_expected_sample_counts = torch.zeros(
            self.num_motions, dtype=torch.long, device=self.device
        )
        self.env_segment_episode_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_failure_stats_update_step = 0
        self._last_hard_window_refresh_step = -self.hard_window_refresh_steps
        self._last_frontier_window_refresh_step = (
            -self.frontier_window_refresh_steps
        )
        self._previous_diagnostic_sampling_probs = uniform.clone()
        self._last_actual_expected_sampling_l1 = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._last_snapshot_step = -1
        rendering_dir = Path(
            getattr(
                self.config,
                "save_rendering_dir",
                Path.cwd() / "renderings_training",
            )
        )
        self._resampling_snapshot_dir = (
            rendering_dir.parent / "resampling_snapshots"
        )

        logger.info(
            f"Initialized 1-second segment resampling with up to "
            f"{self.max_num_time_segments} segments per motion "
            f"(counts={self.motion_segment_counts.tolist()}, "
            f"distribution-aware={self.enable_distribution_aware_resampling}, "
            f"failure-aware={self.enable_failure_aware_resampling})"
        )

    def _load_distribution_resampling_weights(self):
        weights_path = getattr(self.config.robot.motion, 'distribution_weights_path', None)
        if weights_path in (None, "", "null"):
            raise ValueError(
                "distribution_weights_path must be set when distribution-aware resampling is enabled"
            )

        weights_path = Path(weights_path)
        if not weights_path.is_absolute():
            weights_path = Path.cwd() / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Distribution weights file not found: {weights_path}")

        with open(weights_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or "motions" not in payload:
            raise ValueError(f"Unsupported distribution weights format in {weights_path}")

        file_segment_duration = payload.get("segment_duration_s")
        is_legacy = file_segment_duration is None
        if is_legacy:
            message = (
                f"Legacy distribution weights at {weights_path} do not contain "
                "segment_duration_s; regenerate them for 1-second resampling."
            )
            if self.strict_distribution_weights:
                raise ValueError(message)
            logger.warning(message)
        elif not np.isclose(float(file_segment_duration), self.segment_duration_s):
            raise ValueError(
                f"distribution weights use segment_duration_s={file_segment_duration}, "
                f"expected {self.segment_duration_s}"
            )

        motion_weights = payload["motions"]
        file_segment_counts = payload.get("motion_segment_counts", {})
        missing_keys = []
        used_keys = set()
        loaded = 0
        weights = masked_uniform(self.motion_segment_valid_mask)

        motion_keys = getattr(self._motion_lib, "_motion_data_keys", [])
        for motion_idx in range(self.num_motions):
            if motion_idx >= len(motion_keys):
                missing_keys.append(str(motion_idx))
                continue
            motion_key = str(motion_keys[motion_idx])
            lookup_key = self._match_distribution_weight_key(motion_key, motion_weights)
            if lookup_key is None:
                missing_keys.append(motion_key)
                continue
            used_keys.add(lookup_key)

            value = motion_weights[lookup_key]
            expected_count = int(self.motion_segment_counts[motion_idx].item())
            declared_count = file_segment_counts.get(lookup_key, len(value))
            if is_legacy and not self.strict_distribution_weights:
                legacy_count = int(payload.get("num_time_segments", len(value)))
                if int(declared_count) != legacy_count or len(value) != legacy_count:
                    raise ValueError(
                        f"Legacy distribution weights for motion '{lookup_key}' "
                        f"declare {declared_count} segments but contain {len(value)}."
                    )
                legacy_weights = torch.tensor(
                    value, dtype=torch.float32, device=self.device
                )
                legacy_indices = torch.clamp(
                    torch.floor(
                        (torch.arange(expected_count, device=self.device) + 0.5)
                        / expected_count
                        * legacy_count
                    ).long(),
                    max=legacy_count - 1,
                )
                value = legacy_weights[legacy_indices]
            elif int(declared_count) != expected_count or len(value) != expected_count:
                raise ValueError(
                    f"Distribution weights for motion '{lookup_key}' have "
                    f"{len(value)} segments (declared {declared_count}), expected {expected_count}."
                )

            segment_weights = torch.as_tensor(
                value, dtype=torch.float32, device=self.device
            )
            if not torch.isfinite(segment_weights).all() or segment_weights.sum() <= 0:
                raise ValueError(
                    f"Distribution weights for motion '{lookup_key}' are invalid."
                )
            weights[motion_idx, :expected_count] = segment_weights
            loaded += 1

        if missing_keys:
            message = (
                f"Distribution weights missing for {len(missing_keys)}/{self.num_motions} motions. "
                f"Examples: {missing_keys[:5]}"
            )
            if self.strict_distribution_weights:
                raise KeyError(message)
            logger.warning(message + "; using uniform weights for missing motions.")
        extra_keys = set(motion_weights) - used_keys
        if extra_keys:
            message = (
                f"Distribution weights contain {len(extra_keys)} unmatched motion keys. "
                f"Examples: {sorted(extra_keys)[:5]}"
            )
            if self.strict_distribution_weights:
                raise KeyError(message)
            logger.warning(message)

        self.distribution_segment_weights = apply_distribution_floor(
            weights,
            self.motion_segment_valid_mask,
            self.distribution_min_uniform_ratio,
        )
        logger.info(f"Loaded distribution-aware segment weights for {loaded}/{self.num_motions} motions from {weights_path}")

    def _match_distribution_weight_key(self, motion_key, motion_weights):
        key_path = Path(motion_key)
        candidates = [
            motion_key,
            key_path.name,
            key_path.stem,
        ]
        if not str(motion_key).endswith(".pkl"):
            candidates.append(f"{motion_key}.pkl")
        for candidate in candidates:
            if candidate in motion_weights:
                return candidate
        return None

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
        if "motion_tracking_selected_keypoints" in self.config.robot.motion:
            self.motion_tracking_selected_keypoints_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_selected_keypoints]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        
        self.end_time_ratio_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)

    def _init_motion_far_termination_config(self):
        self.terminate_when_motion_far_body_ids = None
        self.terminate_when_motion_far_axis_ids = None
        if not self.config.termination.terminate_when_motion_far:
            return

        body_ids, axis_ids = resolve_motion_far_termination_config(
            self.config.termination,
            self.simulator._body_list,
            self.device,
        )
        self.terminate_when_motion_far_body_ids = body_ids
        self.terminate_when_motion_far_axis_ids = axis_ids
        
    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs, 
                                         self.num_bodies + self.num_extend_bodies, 
                                         3, 
                                         dtype=torch.float, 
                                         device=self.device, 
                                         requires_grad=False) # extend
            
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def start_compute_metrics(self):
        self.compute_metrics = True
        self.start_idx = 0
    
    def forward_motion_samples(self):
        pass
    
    def _init_buffers(self):
        super()._init_buffers()
        self._init_adaptive_tracking_sigma()
        self.vr_3point_marker_coords = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.realtime_vr_keypoints_pos = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.realtime_vr_keypoints_vel = torch.zeros(3, 3, dtype=torch.float, device=self.device, requires_grad=False) # hand, hand, head
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        
    def _init_adaptive_tracking_sigma(self):
        if not 'adaptive_tracking_sigma' in self.config.rewards or not self.config.rewards.adaptive_tracking_sigma.enable:
            self._update_adaptive_tracking_sigma = lambda *args, **kwargs: None
            logger.info("Adaptive tracking sigma is disabled")
            return
        logger.info("Adaptive tracking sigma is enabled")
        self._reward_tracking_sigma_ema = dict()
        for key, value in self.config.rewards.reward_tracking_sigma.items():
            self._reward_tracking_sigma_ema[key] = value

    def _update_adaptive_tracking_sigma(self, error: torch.Tensor, term: str):
        alpha = self.config.rewards.adaptive_tracking_sigma.alpha
        scale = self.config.rewards.adaptive_tracking_sigma.scale
        adptive_type = self.config.rewards.adaptive_tracking_sigma.type

        self._reward_tracking_sigma_ema[term] = self._reward_tracking_sigma_ema[term] * (1 - alpha) + error.mean().item() * alpha
        if adptive_type == 'origin':
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_tracking_sigma_ema[term], self.config.rewards.reward_tracking_sigma[term])
        elif adptive_type == 'mean':
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_tracking_sigma_ema[term], (self.config.rewards.reward_tracking_sigma[term] + self._reward_tracking_sigma_ema[term]) / 2)
        elif adptive_type == 'scale':
            self.config.rewards.reward_tracking_sigma[term] = min(self._reward_tracking_sigma_ema[term] * scale, self.config.rewards.reward_tracking_sigma[term])
    
    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _reset_tasks_callback(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_tasks_callback(env_ids)
        
        end_time = self.last_episode_length_buf[env_ids]*self.dt + self.motion_start_times[env_ids]
        self.end_time_ratio_buf[env_ids] = (end_time / self.motion_len[env_ids])
        self.log_dict["end_time_ratio"] = self.end_time_ratio_buf.mean()
        self.log_dict["end_time_ratio_std"] = self.end_time_ratio_buf.std()

        # Update segment resampling statistics before choosing the next start time.
        if self.enable_segment_resampling:
            completed_episode_mask = eligible_episode_mask(
                self.env_segment_episode_active[env_ids],
                self.last_episode_length_buf[env_ids],
            )
            completed_env_ids = env_ids[completed_episode_mask]
            if len(completed_env_ids) > 0:
                failed_mask = (
                    self.reset_buf[completed_env_ids].bool()
                    & ~self.time_out_buf[completed_env_ids].bool()
                )
                self._update_motion_failure_stats(
                    completed_env_ids, failed_mask
                )
            self.env_segment_episode_active[env_ids] = False
        
        self._resample_motion_times(env_ids) # need to resample before reset root states
        curriculum_frozen = self._curriculum_is_frozen()
        if self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum and not curriculum_frozen:
            self._update_terminate_when_motion_far_curriculum()
        
        if self.config.termination.terminate_when_dof_far and self.config.termination_curriculum.terminate_when_dof_far_curriculum and not curriculum_frozen:
            self._update_terminate_when_dof_far_curriculum()
        
        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable and self.config.soft_dynamic_correction.curriculum.enable and not curriculum_frozen:
            self._update_soft_dynamic_correction()
    
    def _update_soft_dynamic_correction(self):
        sdc_curr = self.config.soft_dynamic_correction.curriculum
        
        if self.average_episode_length < sdc_curr.level_down_threshold:
            self.sdc_alpha *= (1 + sdc_curr.degree)
        elif self.average_episode_length > sdc_curr.level_up_threshold:
            self.sdc_alpha *= (1 - sdc_curr.degree)
        self.sdc_alpha = np.clip(self.sdc_alpha, sdc_curr.min_alpha, sdc_curr.max_alpha)
    
    def _update_terminate_when_dof_far_curriculum(self):
        assert self.config.termination.terminate_when_dof_far and self.config.termination_curriculum.terminate_when_dof_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_dof_far_curriculum_level_down_threshold:
            self.terminate_when_dof_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_dof_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_dof_far_curriculum_level_up_threshold:
            self.terminate_when_dof_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_dof_far_curriculum_degree)
        self.terminate_when_dof_far_threshold = np.clip(self.terminate_when_dof_far_threshold, 
                                                        self.config.termination_curriculum.terminate_when_dof_far_threshold_min, 
                                                        self.config.termination_curriculum.terminate_when_dof_far_threshold_max)
    
    def _update_terminate_when_motion_far_curriculum(self):
        assert self.config.termination.terminate_when_motion_far and self.config.termination_curriculum.terminate_when_motion_far_curriculum
        if self.average_episode_length < self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
            self.terminate_when_motion_far_threshold *= (1 + self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        elif self.average_episode_length > self.config.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
            self.terminate_when_motion_far_threshold *= (1 - self.config.termination_curriculum.terminate_when_motion_far_curriculum_degree)
        self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                         self.config.termination_curriculum.terminate_when_motion_far_threshold_max)
        

    def _update_tasks_callback(self):
        super()._update_tasks_callback()
        if self.enable_segment_resampling:
            self._maybe_update_failure_ema()
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()

    def set_is_evaluating(self):
        if self.enable_segment_resampling:
            self._maybe_update_failure_ema(force=True)
        super().set_is_evaluating()
        if self.enable_segment_resampling:
            self.env_segment_episode_active[:] = False

    def reset_all(self):
        if hasattr(self, "env_segment_episode_active"):
            if self.common_step_counter > 0:
                self._maybe_update_failure_ema(force=True)
            self.env_segment_episode_active[:] = False
        return super().reset_all()

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        if (
            hasattr(self, "env_segment_episode_active")
            and (target_states is not None or target_buf is not None)
        ):
            self.env_segment_episode_active[env_ids] = False
        return super().reset_envs_idx(env_ids, target_states, target_buf)

    def _update_reset_buf(self):
        super()._update_reset_buf()
        if self.config.termination.terminate_by_orientation_error:
            ref_root_rot = self.kick_motion_res()['root_rot']
            # Current base orientation
            cur_root_rot = self.base_quat
            # Calculate the relative rotation quaternion: q_rel = q_ref * q_cur^-1
            q_cur_inv = quat_conjugate(cur_root_rot, w_last=True)
            q_rel = quat_mul(ref_root_rot, q_cur_inv, w_last=True)
            # For small angles, 2 * norm(vector_part) is a good approximation of the angle in radians.
            orientation_error_rad = 2 * torch.norm(q_rel[:, :2], dim=-1)
            threshold = self.config.termination_scales.termination_orientation_error
            self.reset_buf_terminate_by["orientation_error"] = orientation_error_rad > threshold
            self.reset_buf |= self.reset_buf_terminate_by["orientation_error"]
        if self.config.termination.terminate_when_motion_far:
            motion_far_errors = compute_motion_far_errors(
                self.dif_global_body_pos,
                self.terminate_when_motion_far_body_ids,
                self.terminate_when_motion_far_axis_ids,
            )
            reset_buf_motion_far = torch.any(motion_far_errors > self.terminate_when_motion_far_threshold, dim=-1)
            self.reset_buf_terminate_by["motion_far"] = reset_buf_motion_far
            self.reset_buf |= reset_buf_motion_far
            # log current motion far threshold
            if self.config.termination_curriculum.terminate_when_motion_far_curriculum:
                self.log_dict["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)
        if self.config.termination.terminate_when_dof_far:
            reset_buf_dof_far = torch.any(torch.norm(self.dif_joint_angles, dim=-1) > self.terminate_when_dof_far_threshold, dim=-1)
            self.reset_buf_terminate_by["dof_far"] = reset_buf_dof_far
            self.reset_buf |= reset_buf_dof_far
            # log current dof far threshold
            if self.config.termination_curriculum.terminate_when_dof_far_curriculum:
                self.log_dict["terminate_when_dof_far_threshold"] = torch.tensor(self.terminate_when_dof_far_threshold, dtype=torch.float)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.reset_buf_terminate_by["motion_end"] = current_time > self.motion_len
            self.time_out_buf |= self.reset_buf_terminate_by["motion_end"]

    def next_task(self):
        # This function is only called when evaluating
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0
        memory_optimize = getattr(self.config.robot.motion, "memory_optimize", True)
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx, memory_optimize=memory_optimize)
        self.reset_all()

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if (
            self.is_evaluating
            and bool(getattr(self.config, "eval_fixed_start_grid", False))
        ):
            interval = float(
                getattr(self.config, "eval_fixed_start_interval_s", 1.0)
            )
            offset = float(
                getattr(self.config, "eval_fixed_start_offset_s", 0.0)
            )
            if interval <= 0:
                raise ValueError(
                    "eval_fixed_start_interval_s must be positive"
                )
            frame_dts = self.motion_dt[self.motion_ids[env_ids]]
            max_start_times = torch.clamp(
                self.motion_len[env_ids] - frame_dts, min=0.0
            )
            grid_times = offset + env_ids.float() * interval
            self.motion_start_times[env_ids] = torch.minimum(
                torch.clamp(grid_times, min=0.0), max_start_times
            )
        elif self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            if self.enable_segment_resampling and not self.is_evaluating:
                self.motion_start_times[env_ids] = self._sample_segment_motion_time(env_ids)
            else:
                motion_lengths = self.motion_len[env_ids]
                motion_indices = self._get_env_motion_indices(env_ids)
                frame_dts = self.unique_motion_frame_dts[motion_indices] if self.enable_segment_resampling else self.motion_dt[self.motion_ids[env_ids]]
                max_start_times = torch.clamp(
                    motion_lengths - frame_dts, min=0.0
                )
                max_start_times = torch.minimum(
                    max_start_times,
                    motion_lengths * self._start_time_curriculum_ratios(motion_lengths),
                )
                self.motion_start_times[env_ids] = (
                    torch.rand(len(env_ids), dtype=torch.float32, device=self.device)
                    * max_start_times
                )
        if self.enable_segment_resampling:
            self.env_segment_episode_active[env_ids] = ~self.is_evaluating
            if self.is_evaluating:
                self.env_hard_window_active[env_ids] = False
                self.env_frontier_window_active[env_ids] = False
                self.env_resampling_pool_ids[env_ids] = 0
        
        self._kick_motion_res_counter = -1

    def _sample_segment_motion_time(self, env_ids):
        """Sample motion times using enabled distribution-aware and failure-aware segment priors."""
        env_ids = env_ids.to(self.device).long()
        env_motion_indices = self._get_env_motion_indices(env_ids)
        sampling_mask, all_ratios = self._current_segment_sampling_mask(
            return_ratios=True
        )
        combined_probs = self._compute_all_segment_sampling_probs(
            sampling_mask
        )
        self._maybe_refresh_hard_windows(sampling_mask)
        if self.hard_window_replay:
            combined_probs = self._contextual_target_probabilities(
                sampling_mask
            )

        segment_probs = combined_probs[env_motion_indices]
        selected_probs = segment_probs.clone()
        segment_idx = torch.multinomial(segment_probs, 1).squeeze(-1)
        hard_active = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        frontier_active = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        failure_prior_active = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        distribution_active = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        uniform_active = torch.ones(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        if (
            self.hard_window_replay
            and self.suppress_immediate_failure_terminal_probability
        ):
            hard_pool_ratio = (
                self.hard_window_ratio * self._current_failure_ramp
            )
            frontier_pool_ratio = (
                self.frontier_window_ratio * self._current_failure_ramp
            )
            distribution_pool_ratio = (
                self._contextual_distribution_pool_ratio()
            )
            failure_pool_ratio = self._contextual_failure_pool_ratio()
            pool_draw = (
                env_ids.float() + 0.5
            ) / float(self.num_envs)
            (
                frontier_active,
                hard_active,
                distribution_active,
                failure_prior_active,
                uniform_active,
            ) = dominant_contextual_sampling_pool_masks(
                pool_draw,
                frontier_pool_ratio,
                hard_pool_ratio,
                distribution_pool_ratio,
                failure_pool_ratio,
            )
            frontier_available = self.frontier_window_mask[
                env_motion_indices
            ].any(dim=1)
            unavailable_frontier = frontier_active & ~frontier_available
            frontier_active = frontier_active & frontier_available
            uniform_active = uniform_active | unavailable_frontier
            if frontier_active.any():
                frontier_probs = self.frontier_window_probabilities[
                    env_motion_indices[frontier_active]
                ]
                selected_probs[frontier_active] = frontier_probs
                segment_idx[frontier_active] = torch.multinomial(
                    frontier_probs, 1
                ).squeeze(-1)
            if hard_active.any():
                hard_probs = self.hard_window_probabilities[
                    env_motion_indices[hard_active]
                ]
                selected_probs[hard_active] = hard_probs
                segment_idx[hard_active] = torch.multinomial(
                    hard_probs, 1
                ).squeeze(-1)
            if distribution_active.any():
                distribution_probs = self._current_distribution_prior[
                    env_motion_indices[distribution_active]
                ]
                selected_probs[distribution_active] = distribution_probs
                segment_idx[distribution_active] = torch.multinomial(
                    distribution_probs, 1
                ).squeeze(-1)
            if failure_prior_active.any():
                failure_probs = self._current_failure_prior[
                    env_motion_indices[failure_prior_active]
                ]
                selected_probs[failure_prior_active] = failure_probs
                segment_idx[failure_prior_active] = torch.multinomial(
                    failure_probs, 1
                ).squeeze(-1)
            if uniform_active.any():
                uniform_probs = masked_uniform(sampling_mask)[
                    env_motion_indices[uniform_active]
                ]
                selected_probs[uniform_active] = uniform_probs
                segment_idx[uniform_active] = torch.multinomial(
                    uniform_probs, 1
                ).squeeze(-1)

        self.env_motion_indices[env_ids] = env_motion_indices
        contextual_active = hard_active | frontier_active
        self.env_hard_window_active[env_ids] = contextual_active
        self.env_frontier_window_active[env_ids] = frontier_active
        pool_ids = torch.zeros(
            len(env_ids), dtype=torch.long, device=self.device
        )
        pool_ids[distribution_active] = 1
        pool_ids[failure_prior_active] = 2
        pool_ids[hard_active & ~frontier_active] = 3
        pool_ids[frontier_active] = 4
        self.env_resampling_pool_ids[env_ids] = pool_ids
        self.env_hard_target_segments[env_ids] = segment_idx
        if contextual_active.any():
            hard_flat_targets = (
                env_motion_indices[contextual_active]
                * self.max_num_time_segments
                + segment_idx[contextual_active]
            )
            self.hard_window_target_counts.view(-1).scatter_add_(
                0,
                hard_flat_targets,
                torch.ones_like(hard_flat_targets),
            )

        motion_lengths = self.motion_len[env_ids]
        frame_dts = self.unique_motion_frame_dts[env_motion_indices]
        segment_starts, segment_ends, max_start_times = segment_time_bounds(
            segment_idx,
            motion_lengths,
            frame_dts,
            self.segment_duration_s,
            all_ratios[env_motion_indices],
        )
        sampled_times = segment_starts + torch.rand(
            len(env_ids), dtype=torch.float32, device=self.device
        ) * torch.clamp(segment_ends - segment_starts, min=0.0)
        if contextual_active.any():
            lookbacks = self.hard_window_dynamic_lookback_s[
                env_motion_indices[contextual_active],
                segment_idx[contextual_active],
            ]
            sampled_times[contextual_active] = hard_window_start_times(
                segment_idx[contextual_active],
                lookbacks,
                self.hard_window_jitter_s,
                self.segment_duration_s,
                max_start_times[contextual_active],
                torch.rand(
                    int(contextual_active.sum().item()),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        sampled_times = torch.minimum(
            torch.clamp(sampled_times, min=0.0), max_start_times
        )
        actual_segments = torch.floor(
            sampled_times / self.segment_duration_s
        ).long()
        actual_segments = torch.minimum(
            actual_segments,
            self.motion_segment_counts[env_motion_indices] - 1,
        )
        self.env_current_segments[env_ids] = actual_segments
        flat_start_idx = (
            env_motion_indices * self.max_num_time_segments + actual_segments
        )
        start_counts = torch.bincount(
            flat_start_idx,
            minlength=self.num_motions * self.max_num_time_segments,
        ).view(self.num_motions, self.max_num_time_segments)
        self.motion_segment_start_counts += start_counts
        diagnostic_mask = ~contextual_active
        if diagnostic_mask.any():
            diagnostic_flat_start_idx = (
                env_motion_indices[diagnostic_mask]
                * self.max_num_time_segments
                + actual_segments[diagnostic_mask]
            )
            self._window_segment_start_counts += torch.bincount(
                diagnostic_flat_start_idx,
                minlength=(
                    self.num_motions * self.max_num_time_segments
                ),
            ).view(self.num_motions, self.max_num_time_segments)
            self._window_expected_probability_mass.index_add_(
                0,
                env_motion_indices[diagnostic_mask],
                selected_probs[diagnostic_mask],
            )
            self._window_expected_sample_counts += torch.bincount(
                env_motion_indices[diagnostic_mask],
                minlength=self.num_motions,
            )
        return sampled_times

    def _maybe_refresh_hard_windows(self, valid_mask, force=False):
        if not self.hard_window_replay:
            return
        refresh_global = force or (
            self.common_step_counter - self._last_hard_window_refresh_step
            >= self.hard_window_refresh_steps
        )
        refresh_frontier = force or (
            self.common_step_counter
            - self._last_frontier_window_refresh_step
            >= self.frontier_window_refresh_steps
        )
        if not refresh_global and not refresh_frontier:
            return
        shrunk_rates, confidence = shrink_failure_rates(
            self.motion_segment_failure_rates,
            self.motion_segment_exposure_counts,
            self.motion_segment_failure_counts,
            valid_mask,
            self.min_samples_for_stats,
            self.failure_confidence_exposure,
        )
        self.hard_window_scores = compute_hard_window_scores(
            shrunk_rates,
            self.motion_segment_predecessor_credit,
            self.motion_segment_learning_progress,
            self.hard_window_immediate_failure_rates,
            valid_mask,
            self.motion_segment_tracking_error_ema
            if self.tracking_aware_hard_window
            else None,
            self.tracking_error_scale,
            self.tracking_quality_floor,
            self.motion_segment_failure_regression,
            self.failure_regression_weight,
        )
        self.hard_window_dynamic_lookback_s = update_hard_window_lookbacks(
            self.hard_window_dynamic_lookback_s,
            self.hard_window_immediate_failure_rates,
            self.hard_window_immediate_exposures,
            valid_mask,
            self.hard_window_immediate_failure_threshold,
            self.segment_duration_s,
            self.hard_window_max_lookback_s,
            self.hard_window_immediate_failure_min_exposures,
        )
        suppression_radius = max(
            int(
                np.ceil(
                    self.hard_window_lookback_s
                    / self.segment_duration_s
                )
            ),
            1,
        )
        if refresh_global:
            minimum_selected = int(
                np.ceil(
                    float(self.motion_segment_counts.max().item())
                    * self.hard_window_ratio
                    / self.hard_window_max_probability_ratio
                )
            )
            self.hard_window_mask = select_hard_windows_nms(
                self.hard_window_scores,
                valid_mask,
                self.hard_window_top_k,
                suppression_radius,
                max(minimum_selected, 1),
            )
            selected_scores = (
                self.hard_window_scores * self.hard_window_mask
            )
            self.hard_window_probabilities = (
                normalize_sparse_hard_window_probabilities(
                    selected_scores,
                    self.hard_window_mask,
                    valid_mask,
                    self.hard_window_ratio,
                    self.hard_window_max_probability_ratio,
                )
            )
            self._last_hard_window_refresh_step = self.common_step_counter

        if refresh_frontier:
            if self.common_step_counter < self.failure_warmup_steps:
                self.frontier_window_mask.zero_()
                self.frontier_anchor_segments.fill_(-1)
                self.frontier_window_probabilities.copy_(
                    masked_uniform(valid_mask)
                )
                self._last_frontier_window_refresh_step = (
                    self.common_step_counter
                )
                return

            (
                per_target_rates,
                mastered_mask,
                impossible_mask,
                learnable_mask,
            ) = classify_frontier_targets(
                self.frontier_window_mask,
                self.frontier_window_attempts,
                self.frontier_window_successes,
                self.frontier_min_attempts,
                self.frontier_mastered_threshold,
                self.frontier_impossible_threshold,
            )
            self._frontier_mastered_mask.copy_(mastered_mask)
            self._frontier_impossible_mask.copy_(impossible_mask)

            impossible_with_context = (
                impossible_mask
                & (
                    self.hard_window_dynamic_lookback_s
                    < self.hard_window_max_lookback_s
                )
            )
            self.hard_window_dynamic_lookback_s = torch.where(
                impossible_with_context,
                torch.clamp(
                    self.hard_window_dynamic_lookback_s
                    + self.segment_duration_s,
                    max=self.hard_window_max_lookback_s,
                ),
                self.hard_window_dynamic_lookback_s,
            )
            impossible_at_max = impossible_mask & ~impossible_with_context

            attempts = (
                self.frontier_window_attempts
                * self.frontier_window_mask
            ).sum(dim=1)
            successes = (
                self.frontier_window_successes
                * self.frontier_window_mask
            ).sum(dim=1)
            aggregate_rates = successes.float() / torch.clamp(
                attempts.float(), min=1.0
            )
            self._last_frontier_success_rate.copy_(aggregate_rates)
            has_frontier = self.frontier_window_mask.any(dim=1)
            selected_attempts = torch.where(
                self.frontier_window_mask,
                self.frontier_window_attempts,
                torch.full_like(
                    self.frontier_window_attempts,
                    self.frontier_min_attempts,
                ),
            )
            selected_rates = torch.where(
                self.frontier_window_mask,
                per_target_rates,
                torch.ones_like(per_target_rates),
            )
            can_advance = (
                has_frontier
                & (
                    selected_attempts.min(dim=1).values
                    >= self.frontier_min_attempts
                )
                & (
                    selected_rates.min(dim=1).values
                    >= self.frontier_advance_success_threshold
                )
            )
            minimum_segments = torch.zeros(
                self.num_motions, dtype=torch.long, device=self.device
            )
            if has_frontier.any():
                selected_ids = torch.arange(
                    self.max_num_time_segments, device=self.device
                ).unsqueeze(0)
                previous_max = (
                    selected_ids
                    * self.frontier_window_mask.long()
                ).max(dim=1).values
                minimum_segments = torch.where(
                    can_advance,
                    previous_max + 1,
                    minimum_segments,
                )
            excluded_mask = mastered_mask | impossible_at_max
            preferred_mask = learnable_mask | impossible_with_context
            new_mask, new_anchors = select_earliest_frontier_windows(
                shrunk_rates,
                confidence,
                self.hard_window_scores,
                valid_mask,
                self.frontier_window_top_k,
                self.frontier_failure_threshold,
                self.frontier_confidence_threshold,
                max(
                    int(
                        np.ceil(
                            self.frontier_span_s
                            / self.segment_duration_s
                        )
                    ),
                    1,
                ),
                suppression_radius,
                minimum_segments,
                excluded_mask=excluded_mask,
                preferred_mask=preferred_mask,
                allow_fallback=False,
                contiguous_selection=self.frontier_earliest_contiguous,
            )
            self.frontier_window_mask = new_mask
            self.frontier_anchor_segments = new_anchors
            selected_scores = (
                self.hard_window_scores * self.frontier_window_mask
            )
            active_rows = self.frontier_window_mask.any(dim=1)
            self.frontier_window_probabilities.copy_(
                masked_uniform(valid_mask)
            )
            if active_rows.any():
                self.frontier_window_probabilities[active_rows] = (
                    normalize_sparse_hard_window_probabilities(
                        selected_scores[active_rows],
                        self.frontier_window_mask[active_rows],
                        valid_mask[active_rows],
                        self.frontier_window_ratio,
                        self.frontier_window_max_probability_ratio,
                    )
                )
            removed_targets = ~self.frontier_window_mask
            self.frontier_window_attempts[removed_targets] = 0
            self.frontier_window_successes[removed_targets] = 0
            self._last_frontier_window_refresh_step = (
                self.common_step_counter
            )

    def _contextual_distribution_pool_ratio(self):
        if (
            self.hard_window_use_combined_distribution_weight
            and self._current_combined_distribution_weight >= 0.0
        ):
            return min(
                self.hard_window_distribution_ratio,
                max(self._current_combined_distribution_weight, 0.0),
            )
        return (
            self.hard_window_distribution_ratio
            * self._current_distribution_ramp
        )

    def _contextual_failure_pool_ratio(self):
        return self.hard_window_failure_ratio * self._current_failure_ramp

    def _contextual_target_probabilities(self, valid_mask):
        hard_ratio = self.hard_window_ratio * self._current_failure_ramp
        frontier_available = self.frontier_window_mask.any(
            dim=1, keepdim=True
        ).float()
        frontier_ratio = (
            self.frontier_window_ratio
            * self._current_failure_ramp
            * frontier_available
        )
        distribution_ratio = self._contextual_distribution_pool_ratio()
        failure_ratio = self._contextual_failure_pool_ratio()
        uniform_ratio = (
            1.0
            - frontier_ratio
            - hard_ratio
            - distribution_ratio
            - failure_ratio
        )
        probabilities = (
            uniform_ratio * masked_uniform(valid_mask)
            + distribution_ratio * self._current_distribution_prior
            + failure_ratio * self._current_failure_prior
            + hard_ratio * self.hard_window_probabilities
            + frontier_ratio * self.frontier_window_probabilities
        )
        probabilities = probabilities * valid_mask
        probabilities = probabilities / torch.clamp(
            probabilities.sum(dim=1, keepdim=True), min=1e-12
        )
        self._current_sampling_probs = probabilities
        return probabilities

    def _start_time_curriculum_ratios(self, motion_lengths):
        ratios = torch.ones_like(motion_lengths, dtype=torch.float32)
        if not bool(
            getattr(self.config.robot.motion, "uniform_start_curriculum", False)
        ):
            return ratios
        average_episode_duration = float(self.average_episode_length) * self.dt
        return torch.where(
            average_episode_duration < 0.8 * motion_lengths,
            torch.full_like(ratios, 0.8),
            ratios,
        )

    def _current_segment_sampling_mask(self, return_ratios=False):
        ratios = self._start_time_curriculum_ratios(
            self.unique_motion_lengths
        )
        max_start_times = torch.minimum(
            torch.clamp(
                self.unique_motion_lengths - self.unique_motion_frame_dts,
                min=0.0,
            ),
            self.unique_motion_lengths * ratios,
        )
        mask = mask_segments_by_max_time(
            self.motion_segment_valid_mask,
            self.segment_duration_s,
            max_start_times,
        )
        if return_ratios:
            return mask, ratios
        return mask

    def _get_env_motion_indices(self, env_ids):
        if hasattr(self._motion_lib, "get_env_motion_ids"):
            motion_ids = self._motion_lib.get_env_motion_ids()[self.motion_ids[env_ids]]
        elif hasattr(self._motion_lib, "_curr_motion_ids") and self._motion_lib._curr_motion_ids is not None:
            motion_ids = self._motion_lib._curr_motion_ids[self.motion_ids[env_ids]]
        else:
            motion_ids = torch.remainder(self.motion_ids[env_ids], self.num_motions)
        return torch.remainder(motion_ids.to(self.device).long(), self.num_motions)

    def _compute_segment_sampling_probs(self, motion_idx):
        """Compute final segment probabilities for one motion."""
        return self._compute_all_segment_sampling_probs()[motion_idx]

    def _combined_distribution_guard_active(self):
        return (
            self.combined_distribution_guard
            and self.enable_distribution_aware_resampling
            and self.enable_failure_aware_resampling
            and self.combined_mode == "mixture"
        )

    def _combined_distribution_disabled(self):
        return (
            self._combined_distribution_guard_active()
            and self.common_step_counter
            < self._combined_guard_disabled_until_step
        )

    def _combined_force_failure_active(self):
        return (
            self._combined_distribution_guard_active()
            and self.common_step_counter
            < self._combined_force_failure_until_step
        )

    def _effective_combined_distribution_weights(self):
        if self._combined_distribution_disabled() or self._combined_force_failure_active():
            return 0.0, 0.0
        return (
            self.combined_distribution_weight,
            self.combined_distribution_weight_final,
        )

    def _maybe_update_combined_distribution_guard(
        self, probabilities, valid_mask
    ):
        if not self._combined_distribution_guard_active():
            return False
        if (
            self.common_step_counter - self._last_combined_guard_step
            < self.combined_guard_refresh_steps
        ):
            return False
        self._last_combined_guard_step = self.common_step_counter

        confidence = self._current_failure_confidence[valid_mask].mean()
        if confidence < self.combined_guard_min_failure_confidence:
            return False

        conflict = distribution_failure_topk_conflict(
            self._current_distribution_prior,
            self._current_failure_prior,
            valid_mask,
            self.combined_guard_topk_fraction,
        )
        mean_conflict = conflict.mean()
        diagnostics = prior_alignment_diagnostics(
            probabilities,
            self._current_distribution_prior,
            self._current_failure_prior,
            valid_mask,
            self.combined_guard_topk_fraction,
        )
        gate_loss = diagnostics["distribution_gate_loss"]
        self._last_distribution_conflict.copy_(mean_conflict)
        self._last_combined_guard_gate_loss.copy_(gate_loss)

        distribution_survival = self._resampling_pool_survival_ema[1]
        competitor_survival = torch.max(
            self._resampling_pool_survival_ema[0],
            torch.max(
                self._resampling_pool_survival_ema[2],
                torch.max(
                    self._resampling_pool_survival_ema[3],
                    self._resampling_pool_survival_ema[4],
                ),
            ),
        )
        has_pool_evidence = (
            self._resampling_pool_counts[1] >= 32
            and (
                self._resampling_pool_counts[0]
                + self._resampling_pool_counts[2]
                + self._resampling_pool_counts[3]
                + self._resampling_pool_counts[4]
                >= 32
            )
        )
        pool_underperforms = bool(
            has_pool_evidence
            and distribution_survival
            + self.combined_guard_pool_survival_margin_s
            < competitor_survival
        )
        prior_conflict = bool(
            mean_conflict >= self.combined_guard_conflict_threshold
            or gate_loss >= self.combined_guard_gate_loss_threshold
        )
        if prior_conflict or pool_underperforms:
            self._combined_guard_disabled_until_step = (
                self.common_step_counter + self.combined_guard_disable_steps
            )
            self._combined_guard_trigger_count += 1
            if pool_underperforms:
                self._combined_force_failure_until_step = (
                    self.common_step_counter
                    + self.combined_force_failure_disable_steps
                )
                self._combined_force_failure_count += 1
            logger.info(
                "Combined distribution guard triggered at step {}: "
                "conflict={:.3f}, gate_loss={:.3f}, confidence={:.3f}, "
                "pool_underperforms={}".format(
                    self.common_step_counter,
                    float(mean_conflict),
                    float(gate_loss),
                    float(confidence),
                    pool_underperforms,
                )
            )
            return True
        return False

    def _compute_all_segment_sampling_probs(self, valid_mask=None):
        """Compute final segment probabilities for all motions."""
        if valid_mask is None:
            valid_mask = self.motion_segment_valid_mask
        distribution_ramp_value = failure_ramp(
            self.common_step_counter,
            self.distribution_warmup_steps,
            self.distribution_ramp_steps,
        )
        failure_ramp_value = failure_ramp(
            self.common_step_counter,
            self.failure_warmup_steps,
            self.failure_ramp_steps,
        )
        combined_handoff_value = failure_ramp(
            self.common_step_counter,
            self.combined_handoff_start_steps,
            self.combined_handoff_ramp_steps,
        )
        (
            probs,
            self._current_distribution_prior,
            self._current_failure_prior,
            self._current_failure_confidence,
            self._current_cap_hit_counts,
        ) = (
            compute_sampling_probabilities(
                valid_mask,
                self.distribution_segment_weights,
                self.motion_segment_failure_rates,
                self.motion_segment_exposure_counts,
            self.motion_segment_failure_counts,
            self.enable_distribution_aware_resampling,
            self.enable_failure_aware_resampling,
                self.min_samples_for_stats,
                self.failure_confidence_exposure,
                self.failure_rate_epsilon,
                self.failure_bias_strength,
                self.distribution_strength,
                self.uniform_mix_ratio,
                self.max_sampling_probability_ratio,
                failure_ramp_value,
                self.combined_mode,
                self.combined_failure_weight,
                distribution_ramp_value,
                self._effective_combined_distribution_weights()[0],
                self.failure_priority_mode,
                combined_handoff_value,
                self._effective_combined_distribution_weights()[1],
                self.combined_failure_weight_final,
            )
        )
        self._current_distribution_ramp = distribution_ramp_value
        self._current_failure_ramp = failure_ramp_value
        self._current_combined_handoff = combined_handoff_value
        if (
            self.enable_distribution_aware_resampling
            and self.enable_failure_aware_resampling
            and self.combined_mode == "mixture"
        ):
            (
                self._current_combined_uniform_weight,
                self._current_combined_distribution_weight,
                self._current_combined_failure_weight,
            ) = combined_mixture_weights(
                distribution_ramp_value,
                failure_ramp_value,
                combined_handoff_value,
                self._effective_combined_distribution_weights()[0],
                self.combined_failure_weight,
                self._effective_combined_distribution_weights()[1],
                self.combined_failure_weight_final,
            )
        else:
            self._current_combined_uniform_weight = -1.0
            self._current_combined_distribution_weight = -1.0
            self._current_combined_failure_weight = -1.0
        if self._combined_force_failure_active():
            (
                probs,
                self._current_distribution_prior,
                self._current_failure_prior,
                self._current_failure_confidence,
                self._current_cap_hit_counts,
            ) = compute_sampling_probabilities(
                valid_mask,
                self.distribution_segment_weights,
                self.motion_segment_failure_rates,
                self.motion_segment_exposure_counts,
                self.motion_segment_failure_counts,
                False,
                True,
                self.min_samples_for_stats,
                self.failure_confidence_exposure,
                self.failure_rate_epsilon,
                self.failure_bias_strength,
                self.distribution_strength,
                self.uniform_mix_ratio,
                self.max_sampling_probability_ratio,
                1.0,
                self.combined_mode,
                self.combined_failure_weight_final,
                0.0,
                0.0,
                self.failure_priority_mode,
                1.0,
                0.0,
                self.combined_failure_weight_final,
            )
            self._current_combined_uniform_weight = self.uniform_mix_ratio
            self._current_combined_distribution_weight = 0.0
            self._current_combined_failure_weight = (
                1.0 - self.uniform_mix_ratio
            )
        elif self._maybe_update_combined_distribution_guard(probs, valid_mask):
            return self._compute_all_segment_sampling_probs(valid_mask)
        if (
            self.hard_window_replay
            and self.suppress_immediate_failure_terminal_probability
        ):
            (
                self._current_failure_prior,
                self.hard_window_terminal_probability_multiplier,
            ) = suppress_immediate_failure_probabilities(
                self._current_failure_prior,
                self.hard_window_immediate_failure_rates,
                self.hard_window_immediate_exposures,
                valid_mask,
                self.hard_window_immediate_failure_threshold,
                self.hard_window_immediate_failure_min_exposures,
                self.hard_window_terminal_suppression_floor,
            )
        self._current_sampling_probs = probs
        return probs

    def _update_motion_failure_stats(self, env_ids, failed_mask):
        """Update per-segment sample counts and failure EMA."""
        if (
            not self.enable_segment_resampling
            or self.is_evaluating
            or len(env_ids) == 0
        ):
            return

        env_ids = env_ids.to(self.device).long()
        failed_mask = failed_mask.to(self.device).bool()
        if failed_mask.shape != env_ids.shape:
            raise ValueError(
                "failed_mask must be aligned one-to-one with env_ids"
            )
        motion_idx = self.env_motion_indices[env_ids]
        start_segments = self.env_current_segments[env_ids]
        terminal_times = (
            self.motion_start_times[env_ids]
            + self.last_episode_length_buf[env_ids].float() * self.dt
        )
        terminal_times = torch.minimum(
            terminal_times,
            torch.clamp(
                self.motion_len[env_ids]
                - self.unique_motion_frame_dts[motion_idx],
                min=0.0,
            ),
        )
        terminal_segments = torch.floor(
            terminal_times / self.segment_duration_s
        ).long()
        terminal_segments = torch.minimum(
            terminal_segments,
            self.motion_segment_counts[motion_idx] - 1,
        )
        terminal_segments = torch.maximum(terminal_segments, start_segments)

        batch_exposures, batch_failures = accumulate_episode_statistics(
            motion_idx,
            start_segments,
            terminal_segments,
            failed_mask,
            self.num_motions,
            self.max_num_time_segments,
        )
        self._pending_segment_exposure_counts += batch_exposures
        self._pending_segment_failure_counts += batch_failures
        lookback_segments = int(
            np.ceil(self.failure_credit_lookback_s / self.segment_duration_s)
        )
        decay_segments = (
            self.failure_credit_decay_s / self.segment_duration_s
        )
        self._pending_segment_predecessor_credit += (
            accumulate_predecessor_failure_credit(
                motion_idx,
                terminal_segments,
                failed_mask,
                self.num_motions,
                self.max_num_time_segments,
                lookback_segments,
                decay_segments,
            )
        )
        if self.tracking_aware_hard_window:
            body_tracking_error = torch.norm(
                self.dif_global_body_pos[env_ids], dim=-1
            ).mean(dim=-1)
            joint_tracking_error = torch.abs(
                self.dif_joint_angles[env_ids]
            ).mean(dim=-1)
            tracking_error = (
                body_tracking_error
                + self.tracking_joint_error_weight * joint_tracking_error
            ).float()
            flat_terminal_segments = (
                motion_idx * self.max_num_time_segments + terminal_segments
            )
            self._pending_segment_tracking_error_sums.view(-1).scatter_add_(
                0, flat_terminal_segments, tracking_error
            )
            self._pending_segment_tracking_error_counts.view(-1).scatter_add_(
                0,
                flat_terminal_segments,
                torch.ones_like(flat_terminal_segments),
            )

        pool_ids = self.env_resampling_pool_ids[env_ids]
        episode_seconds = self.last_episode_length_buf[env_ids].float() * self.dt
        success_values = (~failed_mask).float()
        pool_counts = torch.bincount(pool_ids, minlength=5).float()
        if pool_counts.sum() > 0:
            survival_sums = torch.zeros(
                5, dtype=torch.float32, device=self.device
            )
            success_sums = torch.zeros(
                5, dtype=torch.float32, device=self.device
            )
            survival_sums.scatter_add_(0, pool_ids, episode_seconds)
            success_sums.scatter_add_(0, pool_ids, success_values)
            observed = pool_counts > 0
            survival_means = survival_sums / torch.clamp(pool_counts, min=1.0)
            success_means = success_sums / torch.clamp(pool_counts, min=1.0)
            alpha = self.failure_rate_ema_alpha
            self._resampling_pool_survival_ema = torch.where(
                observed,
                (1.0 - alpha) * self._resampling_pool_survival_ema
                + alpha * survival_means,
                self._resampling_pool_survival_ema,
            )
            self._resampling_pool_success_ema = torch.where(
                observed,
                (1.0 - alpha) * self._resampling_pool_success_ema
                + alpha * success_means,
                self._resampling_pool_success_ema,
            )
            self._resampling_pool_counts += pool_counts.long()

        hard_active = self.env_hard_window_active[env_ids]
        if hard_active.any():
            hard_motion_idx = motion_idx[hard_active]
            hard_targets = self.env_hard_target_segments[env_ids][hard_active]
            immediate_failed = (
                failed_mask[hard_active]
                & (
                    self.last_episode_length_buf[env_ids][hard_active].float()
                    * self.dt
                    <= self.hard_window_immediate_failure_s
                )
            )
            flat_targets = (
                hard_motion_idx * self.max_num_time_segments + hard_targets
            )
            self.hard_window_immediate_exposures.view(-1).scatter_add_(
                0, flat_targets, torch.ones_like(flat_targets)
            )
            self.hard_window_immediate_failures.view(-1).scatter_add_(
                0,
                flat_targets,
                immediate_failed.long(),
            )

        frontier_active = self.env_frontier_window_active[env_ids]
        if frontier_active.any():
            frontier_motion_idx = motion_idx[frontier_active]
            frontier_targets = self.env_hard_target_segments[env_ids][
                frontier_active
            ]
            frontier_terminals = terminal_segments[frontier_active]
            crossing_margin_segments = max(
                int(
                    np.ceil(
                        self.frontier_crossing_margin_s
                        / self.segment_duration_s
                    )
                ),
                1,
            )
            crossed_target = frontier_crossing_success(
                frontier_targets,
                frontier_terminals,
                failed_mask[frontier_active],
                self.motion_segment_counts[frontier_motion_idx],
                crossing_margin_segments,
            )
            flat_targets = (
                frontier_motion_idx * self.max_num_time_segments
                + frontier_targets
            )
            self.frontier_window_attempts.view(-1).scatter_add_(
                0, flat_targets, torch.ones_like(flat_targets)
            )
            self.frontier_window_successes.view(-1).scatter_add_(
                0, flat_targets, crossed_target.long()
            )

    def _maybe_update_failure_ema(self, force=False):
        if not self.enable_failure_aware_resampling:
            return
        if not should_update_failure_statistics(
            self.common_step_counter,
            self._last_failure_stats_update_step,
            self.failure_stats_update_interval_steps,
            force,
        ):
            return
        if torch.any(self._pending_segment_exposure_counts > 0):
            previous_rates = self.motion_segment_failure_rates.clone()
            previously_observed = self.motion_segment_exposure_counts > 0
            self.motion_segment_failure_rates = update_failure_ema(
                self.motion_segment_failure_rates,
                self._pending_segment_exposure_counts,
                self._pending_segment_failure_counts,
                self.failure_rate_ema_alpha,
            )
            self.motion_segment_exposure_counts += (
                self._pending_segment_exposure_counts
            )
            self.motion_segment_failure_counts += (
                self._pending_segment_failure_counts
            )
            self.motion_segment_predecessor_credit += (
                self._pending_segment_predecessor_credit
            )
            self.motion_segment_learning_progress = torch.clamp(
                previous_rates - self.motion_segment_failure_rates,
                min=0.0,
            )
            self.motion_segment_failure_regression = torch.clamp(
                self.motion_segment_failure_rates - previous_rates,
                min=0.0,
            ) * previously_observed
            self._previous_failure_rates.copy_(previous_rates)
            self.hard_window_immediate_failure_rates = (
                self.hard_window_immediate_failures.float()
                / torch.clamp(
                    self.hard_window_immediate_exposures.float(), min=1.0
                )
            )
            pending_tracking_observed = (
                self._pending_segment_tracking_error_counts > 0
            )
            if pending_tracking_observed.any():
                pending_tracking_means = (
                    self._pending_segment_tracking_error_sums
                    / torch.clamp(
                        self._pending_segment_tracking_error_counts.float(),
                        min=1.0,
                    )
                )
                initialized_tracking = (
                    self.motion_segment_tracking_error_counts > 0
                )
                alpha = self.tracking_error_ema_alpha
                updated_tracking = torch.where(
                    initialized_tracking,
                    (1.0 - alpha) * self.motion_segment_tracking_error_ema
                    + alpha * pending_tracking_means,
                    pending_tracking_means,
                )
                self.motion_segment_tracking_error_ema = torch.where(
                    pending_tracking_observed,
                    updated_tracking,
                    self.motion_segment_tracking_error_ema,
                )
                self.motion_segment_tracking_error_counts += (
                    self._pending_segment_tracking_error_counts
                )
            self._pending_segment_exposure_counts.zero_()
            self._pending_segment_failure_counts.zero_()
            self._pending_segment_predecessor_credit.zero_()
            self._pending_segment_tracking_error_sums.zero_()
            self._pending_segment_tracking_error_counts.zero_()
        self._last_failure_stats_update_step = self.common_step_counter

    def _get_segment_from_time(self, motion_time, motion_length):
        """Get the segment index for a given time in a motion"""
        if motion_length <= 0:
            return 0
        segment_idx = int(motion_time / self.segment_duration_s)
        segment_count = max(int(np.ceil(motion_length / self.segment_duration_s)), 1)
        return min(segment_idx, segment_count - 1)

    def _log_segment_resampling_stats(self):
        """Log statistics about segment resampling."""
        if not self.enable_segment_resampling:
            return

        for key in list(self.log_dict):
            if key.startswith("resampling/motion_"):
                del self.log_dict[key]

        sampling_mask = self._current_segment_sampling_mask()
        probabilities = self._compute_all_segment_sampling_probs(sampling_mask)
        self._maybe_refresh_hard_windows(sampling_mask)
        if self.hard_window_replay:
            probabilities = self._contextual_target_probabilities(
                sampling_mask
            )
        summary = sampling_summary(
            probabilities,
            sampling_mask,
            self._previous_diagnostic_sampling_probs,
        )
        prior_diagnostics = prior_alignment_diagnostics(
            probabilities,
            self._current_distribution_prior,
            self._current_failure_prior,
            sampling_mask,
        )
        self._previous_diagnostic_sampling_probs.copy_(probabilities)

        if self.common_step_counter % self.resampling_log_interval_steps == 0:
            expected_counts = torch.clamp(
                self._window_expected_sample_counts.float().unsqueeze(1),
                min=1.0,
            )
            expected_ratios = (
                self._window_expected_probability_mass / expected_counts
            )
            actual_counts = self._window_segment_start_counts.sum(
                dim=1, keepdim=True
            ).float()
            actual_ratios = (
                self._window_segment_start_counts.float()
                / torch.clamp(actual_counts, min=1.0)
            )
            sampled_motions = (
                self._window_expected_sample_counts > 0
            )
            if sampled_motions.any():
                self._last_actual_expected_sampling_l1 = torch.abs(
                    actual_ratios[sampled_motions]
                    - expected_ratios[sampled_motions]
                ).sum(dim=1).mean()
            else:
                self._last_actual_expected_sampling_l1.zero_()
            self._window_segment_start_counts.zero_()
            self._window_expected_probability_mass.zero_()
            self._window_expected_sample_counts.zero_()

        for name, value in summary.items():
            self.log_dict[f"resampling/{name}"] = value
        for name, value in prior_diagnostics.items():
            self.log_dict[f"resampling/{name}"] = value
        self.log_dict["resampling/uniform_mix_ratio"] = torch.tensor(
            self.uniform_mix_ratio, dtype=torch.float32, device=self.device
        )
        self.log_dict["resampling/distribution_strength"] = torch.tensor(
            self.distribution_strength, dtype=torch.float32, device=self.device
        )
        self.log_dict["resampling/combined_failure_weight"] = torch.tensor(
            self.combined_failure_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict["resampling/combined_distribution_weight"] = torch.tensor(
            self.combined_distribution_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict[
            "resampling/combined_handoff_coefficient"
        ] = torch.tensor(
            self._current_combined_handoff,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict[
            "resampling/active_uniform_weight"
        ] = torch.tensor(
            self._current_combined_uniform_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict[
            "resampling/active_distribution_weight"
        ] = torch.tensor(
            self._current_combined_distribution_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict[
            "resampling/active_failure_weight"
        ] = torch.tensor(
            self._current_combined_failure_weight,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict["resampling/combined_mode_is_mixture"] = torch.tensor(
            1.0 if self.combined_mode == "mixture" else 0.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict["resampling/failure_priority_is_frontier"] = torch.tensor(
            1.0 if self.failure_priority_mode == "frontier" else 0.0,
            dtype=torch.float32,
            device=self.device,
        )
        valid_regression = self.motion_segment_failure_regression[sampling_mask]
        self.log_dict["resampling/failure_regression_mean"] = (
            valid_regression.mean()
        )
        valid_terminal_multiplier = (
            self.hard_window_terminal_probability_multiplier[sampling_mask]
        )
        self.log_dict["resampling/terminal_suppression_count"] = (
            valid_terminal_multiplier < 1.0
        ).sum().float()
        self.log_dict["resampling/terminal_suppression_enabled"] = torch.tensor(
            1.0
            if self.suppress_immediate_failure_terminal_probability
            else 0.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict["resampling/terminal_probability_multiplier_min"] = (
            valid_terminal_multiplier.min()
        )
        self.log_dict["resampling/tracking_aware_hard_window"] = torch.tensor(
            1.0 if self.tracking_aware_hard_window else 0.0,
            dtype=torch.float32,
            device=self.device,
        )
        observed_tracking = self.motion_segment_tracking_error_counts > 0
        observed_valid_tracking = observed_tracking & sampling_mask
        if observed_valid_tracking.any():
            self.log_dict["resampling/tracking_error_ema_mean"] = (
                self.motion_segment_tracking_error_ema[
                    observed_valid_tracking
                ].mean()
            )
            self.log_dict["resampling/tracking_error_ema_max"] = (
                self.motion_segment_tracking_error_ema[
                    observed_valid_tracking
                ].max()
            )
        self.log_dict["resampling/distribution_ramp_coefficient"] = torch.tensor(
            self._current_distribution_ramp,
            dtype=torch.float32,
            device=self.device,
        )
        self.log_dict["resampling/failure_ramp_coefficient"] = torch.tensor(
            self._current_failure_ramp, dtype=torch.float32, device=self.device
        )
        self.log_dict["resampling/combined_distribution_guard_active"] = (
            torch.tensor(
                1.0 if self._combined_distribution_disabled() else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_dict["resampling/combined_force_failure_active"] = (
            torch.tensor(
                1.0 if self._combined_force_failure_active() else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_dict["resampling/combined_guard_trigger_count"] = (
            torch.tensor(
                float(self._combined_guard_trigger_count),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.log_dict["resampling/combined_guard_topk_conflict"] = (
            self._last_distribution_conflict.clone()
        )
        self.log_dict["resampling/combined_guard_gate_loss"] = (
            self._last_combined_guard_gate_loss.clone()
        )
        pool_names = ("uniform", "distribution", "failure", "hard", "frontier")
        for pool_id, pool_name in enumerate(pool_names):
            self.log_dict[
                f"resampling/pool_{pool_name}_survival_ema_s"
            ] = self._resampling_pool_survival_ema[pool_id].clone()
            self.log_dict[
                f"resampling/pool_{pool_name}_success_ema"
            ] = self._resampling_pool_success_ema[pool_id].clone()
            self.log_dict[f"resampling/pool_{pool_name}_count"] = (
                self._resampling_pool_counts[pool_id].float()
            )
        self.log_dict["resampling/cap_hit_count"] = (
            self._current_cap_hit_counts.float().sum()
        )
        self.log_dict["resampling/actual_expected_sampling_l1"] = (
            self._last_actual_expected_sampling_l1.clone()
        )
        if self.hard_window_replay:
            valid_hard = self.hard_window_mask
            hard_count = valid_hard.sum()
            self.log_dict["resampling/hard_window_count"] = (
                hard_count.float()
            )
            self.log_dict["resampling/hard_window_sampling_mass"] = torch.tensor(
                self.hard_window_ratio * self._current_failure_ramp,
                dtype=torch.float32,
                device=self.device,
            )
            self.log_dict["resampling/failure_prior_pool_sampling_mass"] = (
                torch.tensor(
                    self._contextual_failure_pool_ratio(),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.log_dict["resampling/distribution_pool_sampling_mass"] = (
                torch.tensor(
                    self._contextual_distribution_pool_ratio(),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.log_dict["resampling/hard_window_env_ratio"] = (
                (
                    self.env_hard_window_active
                    & ~self.env_frontier_window_active
                ).float().mean()
            )
            self.log_dict["resampling/frontier_window_sampling_mass"] = (
                torch.tensor(
                    self.frontier_window_ratio
                    * self._current_failure_ramp,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.log_dict["resampling/frontier_window_env_ratio"] = (
                self.env_frontier_window_active.float().mean()
            )
            self.log_dict["resampling/frontier_window_count"] = (
                self.frontier_window_mask.sum().float()
            )
            self.log_dict["resampling/frontier_mastered_count"] = (
                self._frontier_mastered_mask.sum().float()
            )
            self.log_dict["resampling/frontier_impossible_count"] = (
                self._frontier_impossible_mask.sum().float()
            )
            valid_anchors = self.frontier_anchor_segments >= 0
            if valid_anchors.any():
                self.log_dict["resampling/frontier_anchor_segment"] = (
                    self.frontier_anchor_segments[
                        valid_anchors
                    ].float().mean()
                )
                self.log_dict["resampling/frontier_success_rate"] = (
                    self._last_frontier_success_rate[
                        valid_anchors
                    ].mean()
                )
            if valid_hard.any():
                self.log_dict["resampling/hard_window_mean_lookback_s"] = (
                    self.hard_window_dynamic_lookback_s[valid_hard].mean()
                )
                self.log_dict[
                    "resampling/hard_window_immediate_failure_rate"
                ] = self.hard_window_immediate_failure_rates[
                    valid_hard
                ].mean()
                self.log_dict[
                    "resampling/hard_window_predecessor_credit"
                ] = self.motion_segment_predecessor_credit[valid_hard].mean()
                self.log_dict[
                    "resampling/hard_window_learning_progress"
                ] = self.motion_segment_learning_progress[valid_hard].mean()
                self.log_dict[
                    "resampling/hard_window_failure_regression"
                ] = self.motion_segment_failure_regression[valid_hard].mean()
                if self.tracking_aware_hard_window:
                    self.log_dict[
                        "resampling/hard_window_tracking_error_ema"
                    ] = self.motion_segment_tracking_error_ema[valid_hard].mean()
                target_indices = torch.nonzero(
                    valid_hard, as_tuple=False
                )[:, 1].float()
                self.log_dict["resampling/hard_window_mean_index"] = (
                    target_indices.mean()
                )
        valid_confidence = self._current_failure_confidence[sampling_mask]
        self.log_dict["resampling/mean_failure_confidence"] = (
            valid_confidence.mean()
        )

        if (
            self.common_step_counter % self.resampling_snapshot_interval_steps
            == 0
            and self.common_step_counter != self._last_snapshot_step
        ):
            self._write_resampling_snapshot(probabilities, sampling_mask)

        if self.common_step_counter % self.resampling_log_interval_steps == 0:
            logger.info(
                "Resampling summary at step {}: entropy={:.4f}, ESS={:.2f}, "
                "max={:.6f}, min_valid={:.6f}, TV={:.4f}, cap_hits={}, "
                "actual_expected_L1={:.4f}, distribution_ramp={:.3f}, "
                "failure_ramp={:.3f}, "
                "corr_dist={:.3f}, corr_fail={:.3f}, gate_loss={:.3f}, "
                "frontier_anchor={:.1f}, frontier_success={:.3f}",
                self.common_step_counter,
                float(summary["entropy"]),
                float(summary["effective_sample_size"]),
                float(summary["max_probability"]),
                float(summary["min_probability"]),
                float(summary["probability_total_variation"]),
                int(self._current_cap_hit_counts.sum().item()),
                float(self._last_actual_expected_sampling_l1),
                self._current_distribution_ramp,
                self._current_failure_ramp,
                float(prior_diagnostics["final_distribution_corr"]),
                float(prior_diagnostics["final_failure_corr"]),
                float(prior_diagnostics["distribution_gate_loss"]),
                float(
                    self.frontier_anchor_segments[
                        self.frontier_anchor_segments >= 0
                    ].float().mean()
                )
                if (
                    self.hard_window_replay
                    and (self.frontier_anchor_segments >= 0).any()
                )
                else -1.0,
                float(self._last_frontier_success_rate.mean())
                if self.hard_window_replay
                else 0.0,
            )

    def _write_resampling_snapshot(self, probabilities, sampling_mask):
        self._resampling_snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = (
            self._resampling_snapshot_dir
            / f"resampling_step_{self.common_step_counter:08d}.npz"
        )
        np.savez_compressed(
            snapshot_path,
            step=np.asarray(self.common_step_counter, dtype=np.int64),
            segment_counts=self.motion_segment_counts.detach().cpu().numpy(),
            valid_mask=sampling_mask.detach().cpu().numpy(),
            final_sampling_probability=probabilities.detach().cpu().numpy(),
            distribution_prior=(
                self._current_distribution_prior.detach().cpu().numpy()
            ),
            failure_prior=self._current_failure_prior.detach().cpu().numpy(),
            failure_confidence=(
                self._current_failure_confidence.detach().cpu().numpy()
            ),
            ema_failure_rate=(
                self.motion_segment_failure_rates.detach().cpu().numpy()
            ),
            exposure_count=(
                self.motion_segment_exposure_counts.detach().cpu().numpy()
            ),
            terminal_failure_count=(
                self.motion_segment_failure_counts.detach().cpu().numpy()
            ),
            predecessor_failure_credit=(
                self.motion_segment_predecessor_credit.detach().cpu().numpy()
            ),
            learning_progress=(
                self.motion_segment_learning_progress.detach().cpu().numpy()
            ),
            failure_regression=(
                self.motion_segment_failure_regression.detach().cpu().numpy()
            ),
            terminal_probability_multiplier=(
                self.hard_window_terminal_probability_multiplier.detach()
                .cpu()
                .numpy()
            ),
            terminal_probability_suppression_enabled=np.asarray(
                self.suppress_immediate_failure_terminal_probability,
                dtype=np.bool_,
            ),
            failure_regression_weight=np.asarray(
                self.failure_regression_weight, dtype=np.float32
            ),
            tracking_error_ema=(
                self.motion_segment_tracking_error_ema.detach().cpu().numpy()
            ),
            tracking_error_count=(
                self.motion_segment_tracking_error_counts.detach().cpu().numpy()
            ),
            tracking_aware_hard_window=np.asarray(
                self.tracking_aware_hard_window, dtype=np.bool_
            ),
            tracking_error_scale=np.asarray(
                self.tracking_error_scale, dtype=np.float32
            ),
            tracking_quality_floor=np.asarray(
                self.tracking_quality_floor, dtype=np.float32
            ),
            hard_window_mask=self.hard_window_mask.detach().cpu().numpy(),
            hard_window_score=(
                self.hard_window_scores.detach().cpu().numpy()
            ),
            hard_window_probability=(
                self.hard_window_probabilities.detach().cpu().numpy()
            ),
            frontier_window_mask=(
                self.frontier_window_mask.detach().cpu().numpy()
            ),
            frontier_window_probability=(
                self.frontier_window_probabilities.detach().cpu().numpy()
            ),
            frontier_anchor_segment=(
                self.frontier_anchor_segments.detach().cpu().numpy()
            ),
            frontier_crossing_margin_s=np.asarray(
                self.frontier_crossing_margin_s, dtype=np.float32
            ),
            frontier_earliest_contiguous=np.asarray(
                self.frontier_earliest_contiguous, dtype=np.bool_
            ),
            frontier_window_attempt_count=(
                self.frontier_window_attempts.detach().cpu().numpy()
            ),
            frontier_window_success_count=(
                self.frontier_window_successes.detach().cpu().numpy()
            ),
            frontier_success_rate=(
                self._last_frontier_success_rate.detach().cpu().numpy()
            ),
            frontier_mastered_mask=(
                self._frontier_mastered_mask.detach().cpu().numpy()
            ),
            frontier_impossible_mask=(
                self._frontier_impossible_mask.detach().cpu().numpy()
            ),
            hard_window_lookback_s=(
                self.hard_window_dynamic_lookback_s.detach().cpu().numpy()
            ),
            hard_window_immediate_failure_rate=(
                self.hard_window_immediate_failure_rates.detach().cpu().numpy()
            ),
            hard_window_immediate_exposure_count=(
                self.hard_window_immediate_exposures.detach().cpu().numpy()
            ),
            hard_window_target_count=(
                self.hard_window_target_counts.detach().cpu().numpy()
            ),
            cumulative_start_count=(
                self.motion_segment_start_counts.detach().cpu().numpy()
            ),
            uniform_mix_ratio=np.asarray(
                self.uniform_mix_ratio, dtype=np.float32
            ),
            distribution_strength=np.asarray(
                self.distribution_strength, dtype=np.float32
            ),
            combined_failure_weight=np.asarray(
                self.combined_failure_weight, dtype=np.float32
            ),
            combined_distribution_weight=np.asarray(
                self.combined_distribution_weight, dtype=np.float32
            ),
            combined_distribution_weight_final=np.asarray(
                self.combined_distribution_weight_final, dtype=np.float32
            ),
            combined_failure_weight_final=np.asarray(
                self.combined_failure_weight_final, dtype=np.float32
            ),
            combined_handoff_coefficient=np.asarray(
                self._current_combined_handoff, dtype=np.float32
            ),
            active_uniform_weight=np.asarray(
                self._current_combined_uniform_weight, dtype=np.float32
            ),
            active_distribution_weight=np.asarray(
                self._current_combined_distribution_weight,
                dtype=np.float32,
            ),
            active_failure_weight=np.asarray(
                self._current_combined_failure_weight, dtype=np.float32
            ),
            contextual_distribution_ratio=np.asarray(
                self._contextual_distribution_pool_ratio()
                if self.hard_window_replay
                else 0.0,
                dtype=np.float32,
            ),
            contextual_failure_ratio=np.asarray(
                self._contextual_failure_pool_ratio()
                if self.hard_window_replay
                else 0.0,
                dtype=np.float32,
            ),
            combined_distribution_guard_active=np.asarray(
                1.0 if self._combined_distribution_disabled() else 0.0,
                dtype=np.float32,
            ),
            combined_force_failure_active=np.asarray(
                1.0 if self._combined_force_failure_active() else 0.0,
                dtype=np.float32,
            ),
            combined_guard_trigger_count=np.asarray(
                self._combined_guard_trigger_count, dtype=np.int64
            ),
            combined_force_failure_count=np.asarray(
                self._combined_force_failure_count, dtype=np.int64
            ),
            combined_guard_topk_conflict=np.asarray(
                float(self._last_distribution_conflict), dtype=np.float32
            ),
            combined_guard_gate_loss=np.asarray(
                float(self._last_combined_guard_gate_loss), dtype=np.float32
            ),
            pool_survival_ema_s=(
                self._resampling_pool_survival_ema.detach().cpu().numpy()
            ),
            pool_success_ema=(
                self._resampling_pool_success_ema.detach().cpu().numpy()
            ),
            pool_counts=self._resampling_pool_counts.detach().cpu().numpy(),
            combined_mode=np.asarray(self.combined_mode),
            failure_priority_mode=np.asarray(self.failure_priority_mode),
            distribution_ramp_coefficient=np.asarray(
                self._current_distribution_ramp, dtype=np.float32
            ),
            failure_ramp_coefficient=np.asarray(
                self._current_failure_ramp, dtype=np.float32
            ),
        )
        self._last_snapshot_step = self.common_step_counter
        logger.info(f"Wrote resampling snapshot to {snapshot_path}")

    def resample_motion(self):
        if self.enable_segment_resampling:
            self._maybe_update_failure_ema(force=True)
        memory_optimize = getattr(self.config.robot.motion, "memory_optimize", True)
        self._motion_lib.load_motions(random_sample=True, memory_optimize=memory_optimize)
        if self.enable_segment_resampling:
            self.env_segment_episode_active[:] = False
        self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))

    def _pre_compute_observations_callback(self):
        super()._pre_compute_observations_callback()
        
        # offset = self.env_origins
        B = self.motion_ids.shape[0]
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        motion_res = self.kick_motion_res()

        if self._motion_lib.has_contact_mask:
            self.ref_contact_mask = motion_res["contact_mask"]
        
        ref_body_pos_extend = motion_res["rg_pos_t"]
        self.ref_body_pos_extend[:] = ref_body_pos_extend # for visualization and analysis
        ref_body_vel_extend = motion_res["body_vel_t"] # [num_envs, num_markers, 3]
        self.ref_body_rot_extend = ref_body_rot_extend = motion_res["rg_rot_t"] # [num_envs, num_markers, 4]
        ref_body_ang_vel_extend = motion_res["body_ang_vel_t"] # [num_envs, num_markers, 3]
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        ref_joint_vel = motion_res["dof_vel"] # [num_envs, num_dofs]


        rotated_pos_in_parent = my_quat_rotate(
            self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([self.simulator._rigid_body_pos, extend_curr_pos], dim=1)

        extend_curr_rot = quat_mul(self.simulator._rigid_body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
                                    self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
                                    w_last=True).view(self.num_envs, -1, 4)
        self._rigid_body_rot_extend = torch.cat([self.simulator._rigid_body_rot, extend_curr_rot], dim=1)
        
        self._rigid_body_ang_vel_extend = torch.cat([self.simulator._rigid_body_ang_vel, self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]], dim=1)
    
        self._rigid_body_ang_vel_global = self.simulator._rigid_body_ang_vel[:, self.extend_body_parent_ids]
        angular_velocity_contribution = torch.cross(self._rigid_body_ang_vel_global, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2)
        extend_curr_vel = self.simulator._rigid_body_vel[:, self.extend_body_parent_ids] + angular_velocity_contribution.view(self.num_envs, -1, 3)
        self._rigid_body_vel_extend = torch.cat([self.simulator._rigid_body_vel, extend_curr_vel], dim=1)

        

        env_batch_size = self.simulator._rigid_body_pos.shape[0]
        num_rigid_bodies = self.simulator._rigid_body_pos.shape[1]

        heading_inv_rot = calc_heading_quat_inv(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        # expand to (B*num_rigid_bodies, 4) for fatser computation in jit
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(1).expand(-1, num_rigid_bodies+self.num_extend_bodies, -1).reshape(-1, 4)

        heading_rot = calc_heading_quat(self.simulator.robot_root_states[:, 3:7].clone(), w_last=True)
        heading_rot_expand = heading_rot.unsqueeze(1).expand(-1, num_rigid_bodies, -1).reshape(-1, 4)

        self.ref_local_root_pos = my_quat_rotate(heading_inv_rot.view(-1, 4), ref_body_pos_extend[:, 0, :].view(-1, 3)).view(env_batch_size, -1, 3)
        self.ref_local_root_rot = get_euler_xyz_in_tensor(quat_mul(heading_inv_rot.view(-1, 4), ref_body_rot_extend[:, 0, :].view(-1, 4), w_last=True).view(env_batch_size, 4))
        self.ref_local_root_vel = my_quat_rotate(heading_inv_rot.view(-1, 4), ref_body_vel_extend[:, 0, :].view(-1, 3)).view(env_batch_size, -1, 3)
        self.ref_local_root_ang_vel = my_quat_rotate(heading_inv_rot.view(-1, 4), ref_body_ang_vel_extend[:, 0, :].view(-1, 3)).view(env_batch_size, -1, 3)
        local_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_pos_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        self.ref_local_foot_pos = local_body_pos[:, self.feet_indices, :]

        self.local_root_pos = my_quat_rotate(heading_inv_rot.view(-1, 4), self.simulator.robot_root_states[:, :3].view(-1, 3)).view(env_batch_size, -1, 3)
        self.local_root_rot = get_euler_xyz_in_tensor(quat_mul(heading_inv_rot.view(-1, 4), self.simulator.robot_root_states[:, 3:7].view(-1, 4), w_last=True).view(env_batch_size, 4))
        local_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), self._rigid_body_pos_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        self.local_foot_pos = local_body_pos[:, self.feet_indices, :]
        


        ## diff compute - kinematic position
        self.dif_global_body_pos = ref_body_pos_extend - self._rigid_body_pos_extend
        self.dif_global_root_pos = ref_body_pos_extend[:, 0, :] - self.simulator.robot_root_states[:, :3]
        ## diff compute - kinematic rotation
        self.dif_global_body_rot = quat_mul(ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend, w_last=True), w_last=True)
        self.dif_global_root_rot = quat_mul(ref_body_rot_extend[:, 0, :], quat_conjugate(self._rigid_body_rot_extend[:, 0, :], w_last=True), w_last=True)
        ref_relative_body_pos = ref_body_pos_extend - ref_body_pos_extend[:, :1, :]
        current_relative_body_pos = self._rigid_body_pos_extend - self._rigid_body_pos_extend[:, :1, :]
        self.dif_relative_body_pos = ref_relative_body_pos - current_relative_body_pos

        ref_root_rot_inv = quat_conjugate(
            ref_body_rot_extend[:, :1, :].expand(-1, ref_body_rot_extend.shape[1], -1),
            w_last=True,
        )
        current_root_rot_inv = quat_conjugate(
            self._rigid_body_rot_extend[:, :1, :].expand(-1, self._rigid_body_rot_extend.shape[1], -1),
            w_last=True,
        )
        ref_relative_body_rot = quat_mul(
            ref_root_rot_inv.reshape(-1, 4),
            ref_body_rot_extend.reshape(-1, 4),
            w_last=True,
        ).view(env_batch_size, -1, 4)
        current_relative_body_rot = quat_mul(
            current_root_rot_inv.reshape(-1, 4),
            self._rigid_body_rot_extend.reshape(-1, 4),
            w_last=True,
        ).view(env_batch_size, -1, 4)
        self.dif_relative_body_rot = quat_mul(
            ref_relative_body_rot,
            quat_conjugate(current_relative_body_rot, w_last=True),
            w_last=True,
        )
        ## diff compute - kinematic velocity
        self.dif_global_body_vel = ref_body_vel_extend - self._rigid_body_vel_extend
        self.dif_global_root_vel = ref_body_vel_extend[:, 0, :] - self.simulator.robot_root_states[:, 7:10]
        ## diff compute - kinematic angular velocity
        self.dif_global_body_ang_vel = ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
        self.dif_global_root_ang_vel = ref_body_ang_vel_extend[:, 0, :] - self.simulator.robot_root_states[:, 10:13]
        # ang_vel_reward = self._reward_teleop_body_ang_velocity_extend()
        

        self.ref_dof_pos = ref_joint_pos
        self.ref_joint_pos = ref_joint_pos - self.default_dof_pos
        self.ref_joint_vel = ref_joint_vel
        ## diff compute - kinematic joint position
        self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
        ## diff compute - kinematic joint velocity
        self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel

        
        # body pos diff in local frame
        self.dif_local_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), self.dif_global_body_pos.view(-1, 3)).view(env_batch_size, -1, 3)
        # root pos diff in local frame
        self.dif_local_root_pos = my_quat_rotate(heading_inv_rot.view(-1, 4), self.dif_global_root_pos.view(-1, 3)).view(env_batch_size, -1, 3)
        # body rot diff in local frame
        self.dif_local_body_rot = quat_mul(heading_inv_rot_expand.view(-1, 4), self.dif_global_body_rot.view(-1, 4), w_last=True).view(env_batch_size, -1, 4)
        # root rot diff in local frame
        self.dif_local_root_rot = quat_mul(heading_inv_rot.view(-1, 4), self.dif_global_root_rot.view(-1, 4), w_last=True).view(env_batch_size, -1, 4)
        # body vel diff in local frame
        self.dif_local_body_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), self.dif_global_body_vel.view(-1, 3)).view(env_batch_size, -1, 3)
        # root vel diff in local frame
        self.dif_local_root_vel = my_quat_rotate(heading_inv_rot.view(-1, 4), self.dif_global_root_vel.view(-1, 3)).view(env_batch_size, -1, 3)
        # body ang vel diff in local frame
        self.dif_local_body_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), self.dif_global_body_ang_vel.view(-1, 3)).view(env_batch_size, -1, 3)
        # root ang vel diff in local frame
        self.dif_local_root_ang_vel = my_quat_rotate(heading_inv_rot.view(-1, 4), self.dif_global_root_ang_vel.view(-1, 3)).view(env_batch_size, -1, 3)

        
        global_ref_rigid_body_pos = ref_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)  # preserves the body position
        local_ref_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_pos.view(-1, 3))
        self._obs_local_ref_rigid_body_pos = local_ref_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)

        global_ref_rigid_body_rot = ref_body_rot_extend.view(env_batch_size, -1, 4)
        local_ref_rigid_body_rot_flat = quat_mul(heading_inv_rot_expand.view(-1, 4), global_ref_rigid_body_rot.view(-1, 4), w_last=True)
        self._obs_local_ref_rigid_body_rot = local_ref_rigid_body_rot_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*4)

        self._obs_local_ref_rigid_body_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        self._obs_local_ref_rigid_body_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_ang_vel_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        
        global_rigid_body_pos = self._rigid_body_pos_extend.view(env_batch_size, -1, 3) - self.simulator.robot_root_states[:, :3].view(env_batch_size, 1, 3)
        local_rigid_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), global_rigid_body_pos.view(-1, 3))
        self._obs_local_rigid_body_pos = local_rigid_body_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        
        global_rigid_body_rot = self._rigid_body_rot_extend.view(env_batch_size, -1, 4)
        local_rigid_body_rot_flat = quat_mul(heading_inv_rot_expand.view(-1, 4), global_rigid_body_rot.view(-1, 4), w_last=True)
        self._obs_local_rigid_body_rot = local_rigid_body_rot_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*4)

        self._obs_local_rigid_body_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), self._rigid_body_vel_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        self._obs_local_rigid_body_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), self._rigid_body_ang_vel_extend.view(-1, 3)).view(env_batch_size, -1, 3)
        
        
        self._obs_dif_local_rigid_body_pos = self.dif_local_body_pos.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
        self._obs_dif_local_rigid_body_rot = self.dif_local_body_rot.view(env_batch_size, -1)
        self._obs_dif_local_rigid_body_vel = self.dif_local_body_vel.view(env_batch_size, -1)
        self._obs_dif_local_rigid_body_ang_vel = self.dif_local_body_ang_vel.view(env_batch_size, -1)

        self._obs_dif_local_root_pos = self.dif_local_root_pos.view(env_batch_size, -1)
        self._obs_dif_local_root_rot = self.dif_local_root_rot.view(env_batch_size, -1)
        self._obs_dif_local_root_vel = self.dif_local_root_vel.view(env_batch_size, -1)
        self._obs_dif_local_root_ang_vel = self.dif_local_root_ang_vel.view(env_batch_size, -1)

        self._obs_local_root_dif_sim = torch.cat([self._obs_dif_local_root_pos, self._obs_dif_local_root_rot, self._obs_dif_local_root_vel, self._obs_dif_local_root_ang_vel], dim=-1)
        self._obs_local_root_dif_real = torch.cat([self._obs_dif_local_root_rot, self._obs_dif_local_root_ang_vel], dim=-1)
        

        # marker_coords for visualization
        self.marker_coords[:] = ref_body_pos_extend.reshape(B, -1, 3)

        if not self.config.use_teleop_control:
            ref_vr_3point_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_id, :]
        else:
            ref_vr_3point_pos = self.teleop_marker_coords
        vr_2root_pos = (ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3))
        heading_inv_rot_vr = heading_inv_rot.repeat(3,1)
        self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr.view(-1, 4), vr_2root_pos.view(-1, 3)).view(env_batch_size, -1)

        self._ref_motion_length = self._motion_lib.get_motion_length(self.motion_ids)
        self._ref_motion_phase = motion_times / self._ref_motion_length
        if not (torch.all(self._ref_motion_phase >= 0) and torch.all(self._ref_motion_phase <= 1.05)): # hard coded 1.05 because +1 will exceed 1
            max_phase = self._ref_motion_phase.max()
        self._ref_motion_phase = self._ref_motion_phase.unsqueeze(1)

        if "motion_tracking_selected_keypoints" in self.config.robot.motion:
            ref_selected_keypoints_pos = ref_body_pos_extend.view(env_batch_size, -1, 3)[:, self.motion_tracking_selected_keypoints_id, :]
            selected_keypoints_2root_pos = ref_selected_keypoints_pos - self.simulator.robot_root_states[:, 0:3].view(env_batch_size, 1, 3)
            heading_inv_rot_selected_keypoints = heading_inv_rot.repeat(len(self.motion_tracking_selected_keypoints_id), 1)
            self._obs_selected_keypoints_pos = my_quat_rotate(heading_inv_rot_selected_keypoints.view(-1, 4), selected_keypoints_2root_pos.view(-1, 3)).view(env_batch_size, -1)
        
        self._log_motion_tracking_info()
    
    def _compute_reward(self):
        super()._compute_reward()
        self.extras["ref_body_pos_extend"] = self.ref_body_pos_extend.clone()
        self.extras["ref_body_rot_extend"] = self.ref_body_rot_extend.clone()

    def _log_motion_tracking_info(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        joint_pos_diff = self.dif_joint_angles

        upper_body_diff_norm = upper_body_diff.norm(dim=-1).mean()
        lower_body_diff_norm = lower_body_diff.norm(dim=-1).mean()
        vr_3point_diff_norm = vr_3point_diff.norm(dim=-1).mean()
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.log_dict["upper_body_diff_norm"] = upper_body_diff_norm
        self.log_dict["lower_body_diff_norm"] = lower_body_diff_norm
        self.log_dict["vr_3point_diff_norm"] = vr_3point_diff_norm
        self.log_dict["joint_pos_diff_norm"] = joint_pos_diff_norm

        if 'enable' in self.config.soft_dynamic_correction and self.config.soft_dynamic_correction.enable:
            self.log_dict["sdc_alpha"] = torch.tensor(self.sdc_alpha, dtype=torch.float, device=self.device)

        if "motion_tracking_selected_keypoints" in self.config.robot.motion:
            selected_keypoints_diff = self.dif_global_body_pos[:, self.motion_tracking_selected_keypoints_id, :]
            selected_keypoints_diff_norm = selected_keypoints_diff.norm(dim=-1).mean()
            self.log_dict["selected_keypoints_diff_norm"] = selected_keypoints_diff_norm
        
        if 'adaptive_tracking_sigma' in self.config.rewards and self.config.rewards.adaptive_tracking_sigma.enable:
            # Create cached tensors for adaptive tracking sigma logging
            for key, value in self._reward_tracking_sigma_ema.items():
                # Cache tensors to avoid repeated creation
                ema_key = f"adaptive_eror_ema_{key}"
                sigma_key = f"adaptive_tracking_sigma_{key}"
                if not hasattr(self, f'_{ema_key}_tensor'):
                    setattr(self, f'_{ema_key}_tensor', torch.tensor(0., dtype=torch.float, device=self.device))
                if not hasattr(self, f'_{sigma_key}_tensor'):
                    setattr(self, f'_{sigma_key}_tensor', torch.tensor(0., dtype=torch.float, device=self.device))
                
                getattr(self, f'_{ema_key}_tensor').fill_(value)
                getattr(self, f'_{sigma_key}_tensor').fill_(self.config.rewards.reward_tracking_sigma[key])
                
                self.log_dict[ema_key] = getattr(self, f'_{ema_key}_tensor')
                self.log_dict[sigma_key] = getattr(self, f'_{sigma_key}_tensor')
        
        self._log_segment_resampling_stats()

    def _draw_debug_vis(self):
        self.simulator.clear_lines()
        self._refresh_sim_tensors()

        for env_id in range(self.num_envs):
            if not self.config.use_teleop_control:
                # draw marker joints
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]): # idx 0 torso (duplicate with 11)
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)

                    self.simulator.draw_sphere(pos_joint, 0.04, color_inner, env_id, pos_id)


            else:
                # draw teleop joints
                for pos_id, pos_joint in enumerate(self.teleop_marker_coords[env_id]):
                    self.simulator.draw_sphere(pos_joint, 0.04, (0.851, 0.144, 0.07), env_id, pos_id)

    def _reset_root_states(self, env_ids):
        # reset root states according to the reference motion
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins: # trimesh
            # offset = self.env_origins
            motion_res = self.kick_motion_res()
            self.simulator.robot_root_states[env_ids, :3] = motion_res['root_pos'][env_ids]
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(motion_res['root_rot'][env_ids])
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = motion_res['root_rot'][env_ids]
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = motion_res['root_vel'][env_ids]
            self.simulator.robot_root_states[env_ids, 10:13] = motion_res['root_ang_vel'][env_ids]
            

        else:
            # offset = self.env_origins
            motion_res = self.kick_motion_res()

            root_pos_noise = self.config.init_noise_scale.root_pos * self.config.noise_to_initial_level
            root_rot_noise = self.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.noise_to_initial_level
            root_vel_noise = self.config.init_noise_scale.root_vel * self.config.noise_to_initial_level
            root_ang_vel_noise = self.config.init_noise_scale.root_ang_vel * self.config.noise_to_initial_level

            root_pos = motion_res['root_pos'][env_ids]
            root_rot = motion_res['root_rot'][env_ids]
            root_vel = motion_res['root_vel'][env_ids]
            root_ang_vel = motion_res['root_ang_vel'][env_ids]

            self.simulator.robot_root_states[env_ids, :3] = root_pos + torch.randn_like(root_pos) * root_pos_noise
            if self.config.simulator.config.name == 'isaacgym':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'isaacsim':
                self.simulator.robot_root_states[env_ids, 3:7] = xyzw_to_wxyz(quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True))
            elif self.config.simulator.config.name == 'genesis':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            elif self.config.simulator.config.name == 'mujoco':
                self.simulator.robot_root_states[env_ids, 3:7] = quat_mul(self.small_random_quaternions(root_rot.shape[0], root_rot_noise), root_rot, w_last=True)
            else:
                raise NotImplementedError
            self.simulator.robot_root_states[env_ids, 7:10] = root_vel + torch.randn_like(root_vel) * root_vel_noise
            self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise


    def small_random_quaternions(self, n, max_angle):
            axis = torch.randn((n, 3), device=self.device)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
            angles = max_angle * torch.rand((n, 1), device=self.device)
            
            # Convert angle-axis to quaternion
            sin_half_angle = torch.sin(angles / 2)
            cos_half_angle = torch.cos(angles / 2)
            
            q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  
            return q

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        # offset = self.env_origins
        motion_res = self.kick_motion_res()

        dof_pos_noise = self.config.init_noise_scale.dof_pos * self.config.noise_to_initial_level
        dof_vel_noise = self.config.init_noise_scale.dof_vel * self.config.noise_to_initial_level
        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        self.simulator.dof_pos[env_ids] = dof_pos + torch.randn_like(dof_pos) * dof_pos_noise
        self.simulator.dof_vel[env_ids] = dof_vel + torch.randn_like(dof_vel) * dof_vel_noise

    _kick_motion_res_counter = -1
    _kick_motion_res_buffer: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter == self.common_step_counter:
            return self._kick_motion_res_buffer # type: ignore
        
        self._kick_motion_res_counter = self.common_step_counter
        
        motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        offset = self.env_origins
        self._kick_motion_res_buffer = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        
        return self._kick_motion_res_buffer
    
    _kick_motion_res_counter_multistep = -1
    _kick_motion_res_multistep_buffer: Optional[Dict[str, torch.Tensor]] = None
    def kick_motion_res_multiplestep(self) -> Dict[str, torch.Tensor]:
        if self._kick_motion_res_counter_multistep == self.common_step_counter:
            return self._kick_motion_res_multiplestep_buffer # type: ignore
        
        self._kick_motion_res_counter_multistep = self.common_step_counter
        buffer = {}

        for i in range(self.config.obs.future_ref_steps):
            motion_times = (self.episode_length_buf + 1 + i) * self.dt + self.motion_start_times
            offset = self.env_origins
            motion= self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
            for k, v in motion.items():
                if k not in buffer:
                    buffer[k] = []
                buffer[k].append(v.unsqueeze(1))

        self._kick_motion_res_multiplestep_buffer = {k: torch.cat(v, dim=1) for k, v in buffer.items()}  # Dict[str, Tensor[B, T, D]]
        return self._kick_motion_res_multiplestep_buffer

    def _post_physics_step(self):
        super()._post_physics_step()
        
        if self.save_motion:    
            motion_times = (self.episode_length_buf) * self.dt + self.motion_start_times

            if (len(self.motions_for_saving['dof'])) > self.config.save_total_steps:
                for k, v in self.motions_for_saving.items():
                    self.motions_for_saving[k] = torch.stack(v[3:]).transpose(0,1).numpy()
                
                self.motions_for_saving['motion_times'] = torch.stack(self.motion_times_buf[3:]).transpose(0,1).numpy()
                
                dump_data = {}
                num_motions = self.num_envs 
                keys_to_save = self.motions_for_saving.keys()

                for i in range(num_motions):
                    motion_key = f"motion{i}" 
                    dump_data[motion_key] = {
                        key: self.motions_for_saving[key][i] for key in keys_to_save
                    }
                    dump_data[motion_key]['fps'] = 1 / self.dt
    
                joblib.dump(dump_data, f'{self.save_motion_dir}.pkl')
                
                print(colored(f"Saved motion data to {self.save_motion_dir}.pkl", 'green'))
                import sys
                sys.exit()

            root_trans = self.simulator.robot_root_states[:, 0:3].cpu()
            if self.config.simulator.config.name == "isaacgym":
                root_rot = self.simulator.robot_root_states[:, 3:7].cpu() # xyzw
            elif self.config.simulator.config.name == "isaacsim":
                root_rot = self.simulator.robot_root_states[:, [4, 5, 6, 3]].cpu() # wxyz to xyzw   
            elif self.config.simulator.config.name == "genesis":
                root_rot = self.simulator.robot_root_states[:,  3:7].cpu() # xyzw
            else:
                raise NotImplementedError
            root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot.numpy()).as_rotvec()).float()
            dof = self.simulator.dof_pos.cpu()
            # T, num_env, J, 3
            pose_aa = torch.cat([root_rot_vec[:, None, :], self._motion_lib.mesh_parsers.dof_axis * dof[:, :, None], torch.zeros((self.num_envs, self.num_augment_joint, 3))], axis = 1)
            self.motions_for_saving['root_trans_offset'].append(root_trans)
            self.motions_for_saving['root_rot'].append(root_rot)
            self.motions_for_saving['dof'].append(dof)
            self.motions_for_saving['pose_aa'].append(pose_aa)
            self.motions_for_saving['action'].append(self.actions.cpu())
            self.motions_for_saving['actor_obs'].append(self.obs_buf_dict['actor_obs'].cpu())
            self.motions_for_saving['terminate'].append(self.reset_buf.cpu())
            
            self.motions_for_saving['dof_vel'].append(self.simulator.dof_vel.cpu())
            self.motions_for_saving['root_lin_vel'].append(self.simulator.robot_root_states[:, 7:10].cpu())
            self.motions_for_saving['root_ang_vel'].append(self.simulator.robot_root_states[:, 10:13].cpu())
            self.motions_for_saving['contact_mask'].append(self.contacts_filt.cpu())
            
            self.motion_times_buf.append(motion_times.cpu())

            self.start_save = True

        
    def _get_obs_dif_local_rigid_body_pos(self):
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_dif_local_rigid_body_rot(self):
        return self._obs_dif_local_rigid_body_rot
    
    def _get_obs_local_ref_rigid_body_rot(self):
        return self._obs_local_ref_rigid_body_rot
    
    def _get_obs_local_ref_rigid_body_vel(self):
        return self._obs_local_ref_rigid_body_vel
    
    def _get_obs_local_ref_rigid_body_ang_vel(self):
        return self._obs_local_ref_rigid_body_ang_vel
    
    def _get_obs_local_root_dif_sim(self):
        return self._obs_local_root_dif_sim
    
    def _get_obs_local_root_dif_real(self):
        return self._obs_local_root_dif_real

    def _get_obs_ref_motion_phase(self):
        return self._ref_motion_phase
    
    def _get_obs_vr_3point_pos(self):
        return self._obs_vr_3point_pos
    
    def _get_obs_selected_keypoints_pos(self):
        return self._obs_selected_keypoints_pos

    def _get_obs_base_pos(self):
        return self.local_root_pos.view(self.num_envs, -1)
    
    def _get_obs_base_rot(self):
        return self.local_root_rot.view(self.num_envs, -1)
    
    def _get_obs_local_foot_pos(self):
        return self.local_foot_pos.view(self.num_envs, -1)
    
    def _get_obs_ref_base_pos(self):
        return self.ref_local_root_pos.view(self.num_envs, -1)
    
    def _get_obs_ref_base_rot(self):
        return self.ref_local_root_rot.view(self.num_envs, -1)
    
    def _get_obs_ref_base_lin_vel(self):
        return self.ref_local_root_vel.view(self.num_envs, -1)
    
    def _get_obs_ref_base_ang_vel(self):
        return self.ref_local_root_ang_vel.view(self.num_envs, -1)

    def _get_obs_ref_local_foot_pos(self):
        return self.ref_local_foot_pos.view(self.num_envs, -1)
    
    def _get_obs_ref_dof_pos(self):
        return self.ref_joint_pos.view(self.num_envs, -1)
    
    def _get_obs_ref_dof_vel(self):
        return self.ref_joint_vel.view(self.num_envs, -1)

    def _get_obs_proprioception(self):
        # proprioception_observation includes:
        # 1. root_augular velocity
        # 2. projected gravity
        # 3. dof position
        # 4. dof velocity
        # 5. actions
        proprio_obs = torch.cat([self.base_ang_vel, self.projected_gravity, self.simulator.dof_pos - self.default_dof_pos, self.simulator.dof_vel, self.actions], dim=1)
        return proprio_obs
    
    def _get_obs_privileged_info(self):
        # privileged_info includes:
        # 1. root linear velocity
        # 2. root height
        # 3. keypoints position
        # 4. foot contact mask
        contact_mask = self._get_obs_feet_contact_mask()
        privileged_info = torch.cat([self.base_lin_vel, self.simulator.robot_root_states[:, 2], self._obs_selected_keypoints_pos, contact_mask], dim=1)
        return privileged_info
    
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
        # for key in history_config.keys():
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    
    def _get_obs_history_critic(self,):
        assert "history_critic" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
        # for key in history_config.keys():
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)

    def _get_obs_history_encoder(self,):
        assert "history_encoder" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_encoder']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
        # for key in history_config.keys():
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1)
    

    def _reward_teleop_body_position_extend(self):
        upper_body_diff = self.dif_global_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight

        self._update_adaptive_tracking_sigma(diff_body_pos_dist_upper, 'teleop_upper_body_pos')
        self._update_adaptive_tracking_sigma(diff_body_pos_dist_lower, 'teleop_lower_body_pos')
    
        return r_body_pos

    def _reward_teleop_body_position_extend_relative(self):
        upper_body_diff = self.dif_relative_body_pos[:, self.upper_body_id, :]
        lower_body_diff = self.dif_relative_body_pos[:, self.lower_body_id, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.config.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.config.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.config.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.config.rewards.teleop_body_pos_upperbody_weight

        self._update_adaptive_tracking_sigma(diff_body_pos_dist_upper, 'teleop_upper_body_pos')
        self._update_adaptive_tracking_sigma(diff_body_pos_dist_lower, 'teleop_lower_body_pos')
    
        return r_body_pos
    
    def _reward_teleop_vr_3point(self):
        vr_3point_diff = self.dif_global_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

        self._update_adaptive_tracking_sigma(vr_3point_dist, 'teleop_vr_3point_pos')
        return r_vr_3point

    def _reward_teleop_vr_3point_relative(self):
        vr_3point_diff = self.dif_relative_body_pos[:, self.motion_tracking_id, :]
        vr_3point_dist = (vr_3point_diff**2).mean(dim=-1).mean(dim=-1)
        r_vr_3point = torch.exp(-vr_3point_dist / self.config.rewards.reward_tracking_sigma.teleop_vr_3point_pos)

        self._update_adaptive_tracking_sigma(vr_3point_dist, 'teleop_vr_3point_pos')
        return r_vr_3point

    def _reward_teleop_selected_keypoints(self):
        selected_keypoints_diff = self.dif_global_body_pos[:, self.motion_tracking_selected_keypoints_id, :]
        selected_keypoints_dist = (selected_keypoints_diff**2).mean(dim=-1).mean(dim=-1)
        r_selected_keypoints = torch.exp(-selected_keypoints_dist / self.config.rewards.reward_tracking_sigma.teleop_selected_keypoints)

        self._update_adaptive_tracking_sigma(selected_keypoints_dist, 'teleop_selected_keypoints')
        return r_selected_keypoints
    
    def _reward_teleop_selected_keypoints_relative(self):
        selected_keypoints_diff = self.dif_relative_body_pos[:, self.motion_tracking_selected_keypoints_id, :]
        selected_keypoints_dist = (selected_keypoints_diff**2).mean(dim=-1).mean(dim=-1)
        r_selected_keypoints = torch.exp(-selected_keypoints_dist / self.config.rewards.reward_tracking_sigma.teleop_selected_keypoints)

        self._update_adaptive_tracking_sigma(selected_keypoints_dist, 'teleop_selected_keypoints')
        return r_selected_keypoints
    
    def _reward_teleop_body_position_feet(self):

        feet_diff = self.dif_global_body_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        self._update_adaptive_tracking_sigma(feet_dist, 'teleop_feet_pos')
        return r_feet

    def _reward_teleop_body_position_feet_relative(self):

        feet_diff = self.dif_relative_body_pos[:, self.feet_indices, :]
        feet_dist = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        r_feet = torch.exp(-feet_dist / self.config.rewards.reward_tracking_sigma.teleop_feet_pos)

        self._update_adaptive_tracking_sigma(feet_dist, 'teleop_feet_pos')
        return r_feet
    
    def _reward_teleop_body_rotation_extend(self):
        rotation_error = compute_quat_angle_error(self.dif_global_body_rot)
        diff_body_rot_dist = (rotation_error ** 2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)

        self._update_adaptive_tracking_sigma(diff_body_rot_dist, 'teleop_body_rot')
        return r_body_rot

    def _reward_teleop_body_rotation_extend_relative(self):
        rotation_error = compute_quat_angle_error(self.dif_relative_body_rot)
        diff_body_rot_dist = (rotation_error ** 2).mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.config.rewards.reward_tracking_sigma.teleop_body_rot)

        self._update_adaptive_tracking_sigma(diff_body_rot_dist, 'teleop_body_rot')
        return r_body_rot

    def _reward_teleop_root_position_global(self):
        root_pos_dist = torch.sum(self.dif_global_root_pos ** 2, dim=-1)
        r_root_pos = torch.exp(-root_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_root_pos)

        self._update_adaptive_tracking_sigma(root_pos_dist, 'teleop_root_pos')
        return r_root_pos

    def _reward_teleop_root_orientation_global(self):
        root_orientation_error = compute_quat_angle_error(self.dif_global_root_rot)
        root_orientation_dist = root_orientation_error ** 2
        r_root_orientation = torch.exp(
            -root_orientation_dist / self.config.rewards.reward_tracking_sigma.teleop_root_orientation
        )

        self._update_adaptive_tracking_sigma(root_orientation_dist, 'teleop_root_orientation')
        return r_root_orientation

    def _reward_teleop_root_orientation(self):
        root_orientation_error = compute_root_orientation_error(self.dif_global_root_rot)
        root_orientation_dist = root_orientation_error ** 2
        r_root_orientation = torch.exp(
            -root_orientation_dist / self.config.rewards.reward_tracking_sigma.teleop_root_orientation
        )

        self._update_adaptive_tracking_sigma(root_orientation_dist, 'teleop_root_orientation')
        return r_root_orientation

    def _reward_teleop_body_velocity_extend(self):
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_vel)

        self._update_adaptive_tracking_sigma(diff_body_vel_dist, 'teleop_body_vel')
        return r_body_vel
    
    def _reward_teleop_body_ang_velocity_extend(self):
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_body_ang_vel)

        self._update_adaptive_tracking_sigma(diff_body_ang_vel_dist, 'teleop_body_ang_vel')
        return r_body_ang_vel

    def _reward_teleop_joint_position(self):
        joint_pos_diff = self.dif_joint_angles
        diff_joint_pos_dist = (joint_pos_diff**2).mean(dim=-1)
        r_joint_pos = torch.exp(-diff_joint_pos_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_pos)

        self._update_adaptive_tracking_sigma(diff_joint_pos_dist, 'teleop_joint_pos')
        return r_joint_pos
    
    def _reward_teleop_joint_velocity(self):
        # # joint_inx_except_under_body = [0, 1, 3, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        joint_vel_diff = self.dif_joint_velocities
        diff_joint_vel_dist = (joint_vel_diff**2).mean(dim=-1)
        r_joint_vel = torch.exp(-diff_joint_vel_dist / self.config.rewards.reward_tracking_sigma.teleop_joint_vel)

        self._update_adaptive_tracking_sigma(diff_joint_vel_dist, 'teleop_joint_vel')
        return r_joint_vel
    
    def _reward_teleop_max_joint_position(self):
        joint_pos_diff:torch.Tensor = self.dif_joint_angles
        max_diff_joint_pos:torch.Tensor = (joint_pos_diff.abs()).max(dim=-1)[0]
        r_max_joint_pos = torch.exp(-max_diff_joint_pos / self.config.rewards.reward_tracking_sigma.teleop_max_joint_pos)
        
        self._update_adaptive_tracking_sigma(max_diff_joint_pos, 'teleop_max_joint_pos')
        return r_max_joint_pos

    def _reward_teleop_contact_mask(self):
        cur_contact_mask = self.contacts_filt.float()
        ref_contact_mask = self.ref_contact_mask.float()
        error_contact_mask = (cur_contact_mask - ref_contact_mask).abs()
        rew = 1 - error_contact_mask.mean(dim=-1)
        return rew
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - self.config.rewards.desired_feet_air_time) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def setup_visualize_entities(self):
        if self.debug_viz and self.config.simulator.config.name == "genesis":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        elif self.debug_viz and self.config.simulator.config.name == "mujoco":
            num_visualize_markers = len(self.config.robot.motion.visualization.marker_joint_colors)
            self.simulator.add_visualize_entities(num_visualize_markers)
        else:
            pass
