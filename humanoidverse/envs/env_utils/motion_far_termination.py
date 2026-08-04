"""Helpers for configurable motion-far termination."""

import torch


AXIS_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
}


def _optional_sequence(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    values = list(value)
    return values if values else None


def _optional_config_value(config, name):
    try:
        if name in config:
            return config[name]
    except TypeError:
        pass
    return getattr(config, name, None)


def resolve_motion_far_body_ids(body_names_config, available_body_names):
    body_names = _optional_sequence(body_names_config)
    if body_names is None:
        return None

    available_body_names = list(available_body_names)
    missing = [name for name in body_names if name not in available_body_names]
    if missing:
        raise ValueError(
            "Invalid terminate_when_motion_far_body_names: "
            f"{missing}. Available bodies: {available_body_names}"
        )
    return [available_body_names.index(name) for name in body_names]


def resolve_motion_far_axis_ids(axes_config):
    axes = _optional_sequence(axes_config)
    if axes is None:
        return None

    normalized_axes = [str(axis).lower() for axis in axes]
    invalid_axes = [axis for axis in normalized_axes if axis not in AXIS_TO_INDEX]
    if invalid_axes:
        raise ValueError(
            "Invalid terminate_when_motion_far_axes: "
            f"{invalid_axes}. Supported axes: {sorted(AXIS_TO_INDEX)}"
        )
    return [AXIS_TO_INDEX[axis] for axis in normalized_axes]


def resolve_motion_far_termination_config(termination_config, available_body_names, device):
    body_names_config = _optional_config_value(
        termination_config,
        "terminate_when_motion_far_body_names",
    )
    axes_config = _optional_config_value(
        termination_config,
        "terminate_when_motion_far_axes",
    )

    body_ids = resolve_motion_far_body_ids(body_names_config, available_body_names)
    axis_ids = resolve_motion_far_axis_ids(axes_config)

    if body_ids is not None:
        body_ids = torch.tensor(body_ids, dtype=torch.long, device=device)
    if axis_ids is not None:
        axis_ids = torch.tensor(axis_ids, dtype=torch.long, device=device)
    return body_ids, axis_ids


def compute_motion_far_errors(dif_global_body_pos, body_ids=None, axis_ids=None):
    dif_pos = dif_global_body_pos
    if body_ids is not None:
        dif_pos = dif_pos[:, body_ids, :]
    if axis_ids is None:
        return torch.norm(dif_pos, dim=-1)

    dif_pos = dif_pos[:, :, axis_ids]
    if dif_pos.shape[-1] == 1:
        return torch.abs(dif_pos.squeeze(-1))
    return torch.norm(dif_pos, dim=-1)


def compute_root_orientation_error(dif_global_root_rot):
    return 2 * torch.norm(dif_global_root_rot[:, :2], dim=-1)


def compute_quat_angle_error(dif_rot):
    quat_vec_norm = torch.norm(dif_rot[..., :3], dim=-1)
    quat_w_abs = torch.abs(dif_rot[..., 3])
    return 2 * torch.atan2(quat_vec_norm, quat_w_abs)
