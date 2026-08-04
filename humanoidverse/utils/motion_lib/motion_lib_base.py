import glob
import os.path as osp
import numpy as np
import joblib
import torch
import random

from humanoidverse.utils.motion_lib.motion_utils.flags import flags
from enum import Enum
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from easydict import EasyDict
from loguru import logger
from rich.progress import track

from isaac_utils.rotations import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
)

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

class MultiMotionFakeCat:
    def __init__(self, unique_motions, motion_assignment, use_fast_single_motion=False):
        """
        Args:
            unique_motions: List of unique motion tensors
            motion_assignment: Tensor[num_envs] - which unique motion each environment uses
            use_fast_single_motion: If True, use torch.cat for single motion scenarios for maximum performance.
                Keep this disabled in the memory-optimized path to avoid duplicating one motion per environment.
        """
        self.unique_motions = unique_motions
        self.motion_assignment = motion_assignment
        self.num_envs = len(motion_assignment)
        
        # Ensure all tensors are on the same device as motion_assignment
        device = motion_assignment.device
        
        # Check if all environments use the same motion (single motion case)
        unique_assignments = torch.unique(motion_assignment)
        self.is_single_motion = len(unique_assignments) == 1
        
        if self.is_single_motion and use_fast_single_motion:
            # Ultra-fast path: use torch.cat for single motion case
            self.single_motion_idx = unique_assignments[0].item()
            single_motion = unique_motions[self.single_motion_idx]
            
            # Convert to tensor and move to correct device if needed
            if isinstance(single_motion, np.ndarray):
                single_motion = torch.from_numpy(single_motion).to(device)
            elif single_motion.device != device:
                single_motion = single_motion.to(device)
            
            # Replicate the motion for all environments (like original torch.cat approach)
            self.cat_tensor = torch.cat([single_motion] * self.num_envs, dim=0)
            self.use_cat_tensor = True
        elif self.is_single_motion:
            # Fast path for single motion case
            self.single_motion_idx = unique_assignments[0].item()
            self.single_motion = unique_motions[self.single_motion_idx]
            
            # Convert to tensor and move to correct device if needed
            if isinstance(self.single_motion, np.ndarray):
                self.single_motion = torch.from_numpy(self.single_motion).to(device)
            elif self.single_motion.device != device:
                self.single_motion = self.single_motion.to(device)
            
            # For single motion case, keep one copy and map global indices back to local frames.
            self.motion_length = len(self.single_motion)
            self.total_length = self.motion_length * self.num_envs
            self.use_cat_tensor = False
        else:
            # Multiple motion case: use original logic
            # Pre-convert all motions to tensors on the correct device
            self.device_motions = []
            for motion in unique_motions:
                if isinstance(motion, np.ndarray):
                    motion = torch.from_numpy(motion).to(device)
                elif motion.device != device:
                    motion = motion.to(device)
                self.device_motions.append(motion)
            
            # the length of each unique motion
            self.unique_motion_lengths = torch.tensor([m.shape[0] for m in self.device_motions], device=device)
            
            # for each environment, which unique motion it uses and that motion's length
            self.env_motion_indices = motion_assignment
            self.env_motion_lengths = self.unique_motion_lengths[motion_assignment]
            
            # the cumulative length for global indexing
            self.env_cumulative_lengths = torch.cat([
                torch.tensor([0], device=device), 
                torch.cumsum(self.env_motion_lengths, dim=0)
            ])
            
            # Precompute index mapping for performance optimization
            self._precompute_index_mapping(device)
            self.use_cat_tensor = False
    
    def _precompute_index_mapping(self, device):
        """Precompute global index to (motion_idx, local_idx) mapping for fast lookup."""
        total_length = self.env_cumulative_lengths[-1].item()
        
        # Only precompute if total length is reasonable (to avoid excessive memory use)
        if total_length <= 1000000:  # 1M frames threshold
            self.motion_idx_map = torch.zeros(total_length, dtype=torch.long, device=device)
            self.local_idx_map = torch.zeros(total_length, dtype=torch.long, device=device)
            
            # Vectorized computation for better performance
            global_indices = torch.arange(total_length, device=device)
            
            # Find which environment each global index belongs to
            env_ids = torch.searchsorted(self.env_cumulative_lengths[1:], global_indices, right=True)
            
            # Calculate local indices in the environment
            local_indices = global_indices - self.env_cumulative_lengths[env_ids]
            local_indices = local_indices % self.env_motion_lengths[env_ids]
            
            # Get the unique motion indices for each environment
            motion_indices = self.env_motion_indices[env_ids]
            
            self.motion_idx_map = motion_indices
            self.local_idx_map = local_indices
            
            self.use_precomputed_mapping = True
        else:
            self.use_precomputed_mapping = False
    
    def __getitem__(self, key):
        # Handle multi-dimensional indexing like [f0l, :]
        if isinstance(key, tuple):
            idx = key[0]
        else:
            idx = key
            
        if isinstance(idx, torch.Tensor):
            # Ensure idx is 1D
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            
            if self.use_cat_tensor:
                # Ultra-fast path: direct tensor indexing
                result = self.cat_tensor[idx]
                
                # If original input was 0D, return single element
                if len(idx) == 1:
                    return result[0]
                else:
                    return result
            elif self.is_single_motion:
                # Fast path for single motion case
                local_indices = idx % self.motion_length
                result = self.single_motion[local_indices]
                
                # If original input was 0D, return single element
                if len(local_indices) == 1:
                    return result[0]
                else:
                    return result
            else:
                # Vectorized logic for multiple motion case
                if hasattr(self, 'use_precomputed_mapping') and self.use_precomputed_mapping:
                    # Ultra-fast path using precomputed mapping
                    # Clamp indices to valid range
                    idx = torch.clamp(idx, 0, len(self.motion_idx_map) - 1)
                    
                    # Direct lookup from precomputed mapping
                    motion_indices = self.motion_idx_map[idx]
                    local_indices = self.local_idx_map[idx]
                    
                    # Group by motion for batch processing
                    unique_motions_in_batch = torch.unique(motion_indices)
                    
                    if len(unique_motions_in_batch) == 1:
                        # All indices belong to the same motion - fastest path
                        motion_idx = unique_motions_in_batch[0].item()
                        result = self.device_motions[motion_idx][local_indices]
                    else:
                        # Multiple motions - use vectorized extraction
                        result_shape = (len(idx),) + self.device_motions[0].shape[1:]
                        result = torch.empty(result_shape, dtype=self.device_motions[0].dtype, device=self.device_motions[0].device)
                        
                        for motion_idx in unique_motions_in_batch:
                            mask = motion_indices == motion_idx
                            if mask.any():
                                motion_local_indices = local_indices[mask]
                                motion_data = self.device_motions[motion_idx.item()][motion_local_indices]
                                result[mask] = motion_data
                else:
                    # Fallback to runtime calculation for very large datasets
                    # determine which environment the index belongs to
                    env_ids = torch.searchsorted(self.env_cumulative_lengths[1:], idx, right=True)
                    
                    # the local index in the environment
                    local_indices = idx - self.env_cumulative_lengths[env_ids]
                    
                    # handle the index out of range (loop access)
                    local_indices = local_indices % self.env_motion_lengths[env_ids]
                    
                    # get the unique motion index for each environment
                    unique_motion_indices = self.env_motion_indices[env_ids]
                    
                    # Vectorized data extraction
                    # Group indices by motion to enable batch processing
                    unique_motions_in_batch = torch.unique(unique_motion_indices)
                    
                    if len(unique_motions_in_batch) == 1:
                        # All indices belong to the same motion - fast path
                        motion_idx = unique_motions_in_batch[0].item()
                        result = self.device_motions[motion_idx][local_indices]
                    else:
                        # Multiple motions involved - use advanced indexing
                        # Pre-allocate result tensor
                        result_shape = (len(idx),) + self.device_motions[0].shape[1:]
                        result = torch.empty(result_shape, dtype=self.device_motions[0].dtype, device=self.device_motions[0].device)
                        
                        # Process each unique motion in batch
                        for motion_idx in unique_motions_in_batch:
                            # Find all indices that belong to this motion
                            mask = unique_motion_indices == motion_idx
                            if mask.any():
                                motion_local_indices = local_indices[mask]
                                motion_data = self.device_motions[motion_idx.item()][motion_local_indices]
                                result[mask] = motion_data
                
                # If original input was 0D, return single element
                if len(idx) == 1:
                    return result[0]
                else:
                    return result
        else:
            # Create tensor on the same device as class tensors
            device = self.motion_assignment.device
            return self[torch.tensor([idx], device=device)][0]
    
    def __len__(self):
        if self.use_cat_tensor:
            return len(self.cat_tensor)
        elif self.is_single_motion:
            return self.total_length
        else:
            if hasattr(self, 'use_precomputed_mapping') and self.use_precomputed_mapping:
                return len(self.motion_idx_map)
            else:
                return self.env_cumulative_lengths[-1].item()
    
    @property
    def shape(self):
        if self.use_cat_tensor:
            return self.cat_tensor.shape
        elif self.is_single_motion:
            base_shape = self.single_motion.shape[1:]
            return torch.Size((self.total_length, *base_shape))
        else:
            base_shape = self.device_motions[0].shape[1:]
            return torch.Size((len(self), *base_shape))

class MotionLibBase():
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        self.has_contact_mask = False
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height = False,  multi_thread = False)
        if flags.real_traj:
            self.track_idx = self._motion_data_load[next(iter(self._motion_data_load))].get("track_idx", [19, 24, 29])
        return
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        self._curr_motion_ids = None
        self._env_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

    def get_env_motion_ids(self):
        """Return raw motion-library ids assigned to each environment."""
        if self._env_motion_ids is None:
            return torch.remainder(
                torch.arange(self.num_envs, dtype=torch.long, device=self._device),
                self._num_unique_motions,
            )
        return self._env_motion_ids

    def get_unique_motion_metadata(self):
        """Return durations and frame intervals indexed by raw motion-library id."""
        lengths = []
        frame_dts = []

        for entry in self._motion_data_list:
            motion = (
                joblib.load(entry)
                if isinstance(entry, (str, Path, np.str_))
                else entry
            )
            if isinstance(motion, dict) and not any(
                key in motion for key in ("dof", "dof_pos", "pose_aa", "root_trans_offset")
            ):
                if len(motion) != 1:
                    raise ValueError(
                        "Directory motion files must contain exactly one motion entry."
                    )
                motion = next(iter(motion.values()))

            fps = float(motion["fps"])
            frame_source = next(
                (
                    motion[key]
                    for key in ("dof", "dof_pos", "pose_aa", "root_trans_offset")
                    if key in motion
                ),
                None,
            )
            if frame_source is None:
                raise KeyError(
                    "Motion entry must contain one of: dof, dof_pos, pose_aa, root_trans_offset"
                )
            num_frames = len(frame_source)
            frame_dt = 1.0 / fps
            lengths.append(max(num_frames - 1, 0) * frame_dt)
            frame_dts.append(frame_dt)

        return (
            torch.tensor(lengths, dtype=torch.float32, device=self._device),
            torch.tensor(frame_dts, dtype=torch.float32, device=self._device),
        )

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action = self._motion_actions[f0l]
        return action

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__: # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel
        
        if self.has_contact_mask:
            contact0, contact1 = self._motion_contact_masks[f0l], self._motion_contact_masks[f1l]
            contact = (1.0 - blend) * contact0 + blend * contact1
            contact = contact.view(contact.shape[0], -1)
            return_dict["contact_mask"] = contact

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "motion_bodies": self._motion_bodies[motion_ids],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
        })
        return return_dict
    
    def load_motions(self, 
                     random_sample=True, 
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None,
                     memory_optimize=None):
        """
        Load motions with configurable memory optimization.
        
        Args:
            memory_optimize (bool or None): 
                - True: Always use memory optimization (slower for multi-motion, faster for single-motion)
                - False: Always use original torch.cat approach (fastest but uses more memory)
                - None: Use memory optimization by default
        """

        # Determine which motions to load for each environment
        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        num_motion_to_load = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)

        # Only process unique motions to avoid duplication
        unique_motion_ids = torch.unique(sample_idxes)
        
        if memory_optimize is None:
            use_memory_optimization = True
            logger.info("Motion memory optimization enabled by default")
        else:
            use_memory_optimization = memory_optimize
            if memory_optimize:
                logger.info("Memory optimization enabled by user")
            else:
                logger.info("Speed optimization enabled by user")
        
        # Use original torch.cat approach if speed optimization is chosen
        if not use_memory_optimization:
            return self._load_motions_with_torch_cat(sample_idxes, target_heading, max_len)
        
        # Continue with memory-optimized approach
        # Create mapping from unique motion ID to index in the unique motions array
        motion_id_to_idx = {motion_id.item(): i for i, motion_id in enumerate(unique_motion_ids)}
        
        # Create motion assignment for each environment
        motion_assignment = torch.tensor([motion_id_to_idx[sample_idx.item()] for sample_idx in sample_idxes], 
                                       device=self._device, dtype=torch.long)
        
        # Process only unique motions
        unique_motions = []
        unique_motion_data = []
        
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        _motion_actions = []
        _motion_contact_masks = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        # Load only unique motions
        for i, motion_id in enumerate(unique_motion_ids):
            motion_data = self._motion_data_list[motion_id.item()]
            res_acc_single = self.load_motion_with_skeleton([motion_data], self.fix_height, target_heading, max_len)
            motion_file_data, curr_motion = res_acc_single[0]
            
            unique_motions.append(curr_motion)
            unique_motion_data.append(motion_file_data)
            
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(torch.from_numpy(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3)).float())
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(torch.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            _motion_lengths.append(curr_len)
            
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.has_contact_mask:
                _motion_contact_masks.append(curr_motion.contact_mask)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion

        # Create lookup tables for motion properties (expanded for all environments)
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)[motion_assignment]
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)[motion_assignment]
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)[motion_assignment]
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)[motion_assignment]
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)[motion_assignment]
        
        # Create MultiMotionFakeCat instances for motion data
        self._motion_aa = MultiMotionFakeCat(_motion_aa, motion_assignment, use_fast_single_motion=False)
        
        if self.has_action:
            self._motion_actions = MultiMotionFakeCat(_motion_actions, motion_assignment, use_fast_single_motion=False)
        if self.has_contact_mask:
            self._motion_contact_masks = MultiMotionFakeCat(_motion_contact_masks, motion_assignment, use_fast_single_motion=False)
            
        # Create MultiMotionFakeCat instances for all motion tensors
        self.gts = MultiMotionFakeCat([m.global_translation for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.grs = MultiMotionFakeCat([m.global_rotation for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.lrs = MultiMotionFakeCat([m.local_rotation for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.grvs = MultiMotionFakeCat([m.global_root_velocity for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.gravs = MultiMotionFakeCat([m.global_root_angular_velocity for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.gavs = MultiMotionFakeCat([m.global_angular_velocity for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.gvs = MultiMotionFakeCat([m.global_velocity for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        self.dvs = MultiMotionFakeCat([m.dof_vels for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        
        # Handle extended motion data if available
        if "global_translation_extend" in unique_motions[0].__dict__:
            self.gts_t = MultiMotionFakeCat([m.global_translation_extend for m in unique_motions], motion_assignment, use_fast_single_motion=False)
            self.grs_t = MultiMotionFakeCat([m.global_rotation_extend for m in unique_motions], motion_assignment, use_fast_single_motion=False)
            self.gvs_t = MultiMotionFakeCat([m.global_velocity_extend for m in unique_motions], motion_assignment, use_fast_single_motion=False)
            self.gavs_t = MultiMotionFakeCat([m.global_angular_velocity_extend for m in unique_motions], motion_assignment, use_fast_single_motion=False)
        
        if "dof_pos" in unique_motions[0].__dict__:
            self.dof_pos = MultiMotionFakeCat([m.dof_pos for m in unique_motions], motion_assignment, use_fast_single_motion=False)
            
        # Handle real trajectory data if available
        if flags.real_traj:
            self.q_gts = MultiMotionFakeCat(self.q_gts, motion_assignment, use_fast_single_motion=False)
            self.q_grs = MultiMotionFakeCat(self.q_grs, motion_assignment, use_fast_single_motion=False)
            self.q_gavs = MultiMotionFakeCat(self.q_gavs, motion_assignment, use_fast_single_motion=False)
            self.q_gvs = MultiMotionFakeCat(self.q_gvs, motion_assignment, use_fast_single_motion=False)
        
        # Calculate length starts for indexing
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        
        # Set up motion tracking
        self._curr_motion_ids = sample_idxes
        self._env_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
        self._num_motions = len(unique_motions)
        self.motion_ids = torch.arange(self._num_motions, dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        
        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Loaded {len(unique_motions)} unique motions (avoiding {num_motion_to_load - len(unique_motions)} duplicates)")
        logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:5]}, ....")
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s.")
        
        return unique_motions

    def _load_motions_with_torch_cat(self, sample_idxes, target_heading, max_len):
        """
        Load motions using original torch.cat approach for maximum speed.
        Uses more memory but provides best performance.
        """
        logger.info("Using torch.cat approach for maximum speed")
        
        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        _motion_actions = []
        _motion_contact_masks = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        # Load motion for each environment (including duplicates)
        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        res_acc = self.load_motion_with_skeleton(motion_data_list, self.fix_height, target_heading, max_len)
        
        for f in track(range(len(res_acc)), description="Loading motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.has_contact_mask:
                _motion_contact_masks.append(curr_motion.contact_mask)
            
            if flags.real_traj:
                self.q_gts.append(curr_motion.quest_motion['quest_trans'])
                self.q_grs.append(curr_motion.quest_motion['quest_rot'])
                self.q_gavs.append(curr_motion.quest_motion['global_angular_vel'])
                self.q_gvs.append(curr_motion.quest_motion['linear_vel'])
                
            del curr_motion

        # Use original torch.cat approach
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.from_numpy(np.concatenate(_motion_aa)).float().to(self._device)
        
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        if self.has_contact_mask:
            self._motion_contact_masks = torch.cat(_motion_contact_masks, dim=0).float().to(self._device)
        
        # Concatenate all motion tensors
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
            
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)
        
        # Calculate length starts for indexing
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        
        # Set up motion tracking
        self._curr_motion_ids = sample_idxes
        self._env_motion_ids = sample_idxes
        self.curr_motion_keys = self._motion_data_keys[sample_idxes.cpu()]
        self._num_motions = len(motions)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        self.num_bodies = self.num_joints
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        
        logger.info(f"Loading {self.num_envs} motions...")
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        
        return motions

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():
            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            
            return trans, height_diff

    def load_motion_with_skeleton(self,
                                  motion_data_list,
                                  fix_height,
                                  target_heading,
                                  max_len):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            if "action" in curr_file.keys():
                self.has_action = True
            if "contact_mask" in curr_file.keys():
                contact_mask_shape = curr_file['contact_mask'].shape
                assert contact_mask_shape[0] == seq_len and len(contact_mask_shape) == 2, "Contact mask shape does not match"
                self._contact_size = contact_mask_shape[1]
                self.has_contact_mask = True
            
            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape

            if not target_heading is None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ])))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))

            if self.mesh_parsers is not None:
                # trans, trans_fix = MotionLibRobot.fix_trans_height(pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
                # add "action" to curr_motion
                if self.has_action:
                    curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
                if self.has_contact_mask:
                    curr_motion.contact_mask = to_torch(curr_file['contact_mask']).clone()[start:end]
                res[f] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
        return res
    

    def num_motions(self):
        return self._num_motions


    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend


    def _get_num_bodies(self):
        return self.num_bodies


    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)
