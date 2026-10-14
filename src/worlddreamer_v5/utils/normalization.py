# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from worlddreamer_v5.data.action_layout import ACTION_DIM, FIELDS, pack_fields, validate_action

from worlddreamer_v5.data.state_layout import STATE_DIM
ACTION_EPS = 1e-6
# Public helper ordering follows the established argument order; field offsets
# come only from the action18 contract (base precedes waist in the tensor).
_PART_NAMES = (
    "left_ee_pos",
    "left_ee_aa",
    "left_gripper",
    "right_ee_pos",
    "right_ee_aa",
    "right_gripper",
    "waist",
    "base",
)
_FIELD_SLICES = {field.name: field.columns for field in FIELDS}
ACTION_PARTS = tuple((name, _FIELD_SLICES[name]) for name in _PART_NAMES)


def get_value(data, path):
    for key in path.split("."):
        if not isinstance(data, Mapping):
            return None
        data = data.get(key)
        if data is None:
            return None
    return data


def resize_image(
    image: Image.Image,
    factor: int = 32,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 160000,
) -> Image.Image:
    width, height = image.size
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")

    new_height = max(factor, round(height / factor) * factor)
    new_width = max(factor, round(width / factor) * factor)

    if new_height * new_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        new_height = math.floor(height / scale / factor) * factor
        new_width = math.floor(width / scale / factor) * factor
    elif new_height * new_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        new_height = math.ceil(height * scale / factor) * factor
        new_width = math.ceil(width * scale / factor) * factor

    return image.resize((new_width, new_height))


def _axis_from_pi(rotm: np.ndarray) -> np.ndarray:
    """Recover a half-turn axis from its largest diagonal component."""
    pivot = int(np.argmax(np.diag(rotm)))
    dominant = np.sqrt(max((rotm[pivot, pivot] + 1) / 2, 0))
    direction = [0.0, 0.0, 0.0]
    direction[pivot] = dominant
    if dominant > 1e-8:
        for other in range(3):
            if other != pivot:
                # Read the upper triangle to preserve the dataset convention.
                direction[other] = rotm[min(pivot, other), max(pivot, other)] / (
                    2 * dominant
                )
    direction = np.array(direction)
    magnitude = np.linalg.norm(direction)
    return np.array([1.0, 0.0, 0.0]) if magnitude < 1e-12 else direction / magnitude


def rotm2aa_batch_stable(rotms: np.ndarray) -> np.ndarray:
    """SO(3) log for new command labels; stable near identity and half turns.

    Project float32 camera/FK products to rotations, then use atan2 rather than
    arccos(trace). The legacy encoder below remains unchanged for old weights.
    """
    matrices = np.asarray(rotms, dtype=np.float64)
    if (
        matrices.ndim != 3
        or matrices.shape[1:] != (3, 3)
        or not np.isfinite(matrices).all()
    ):
        raise ValueError("expected finite rotation matrices [N,3,3]")
    u, _, vt = np.linalg.svd(matrices)
    u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
    matrices = u @ vt
    sine_axis = 0.5 * np.stack(
        [
            matrices[:, 2, 1] - matrices[:, 1, 2],
            matrices[:, 0, 2] - matrices[:, 2, 0],
            matrices[:, 1, 0] - matrices[:, 0, 1],
        ],
        axis=1,
    )
    sine = np.linalg.norm(sine_axis, axis=1)
    cosine = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1) * 0.5, -1, 1)
    theta = np.arctan2(sine, cosine)
    result = sine_axis.copy()
    normal = (theta > 1e-7) & (np.pi - theta > 1e-4)
    result[normal] *= (theta[normal] / sine[normal])[:, None]
    for index in np.flatnonzero(np.pi - theta <= 1e-4):
        _, axes = np.linalg.eigh((matrices[index] + matrices[index].T) * 0.5)
        axis = axes[:, -1]
        if axis @ sine_axis[index] < 0:
            axis = -axis
        result[index] = axis * theta[index]
    return result


def rotm2aa_batch(rotms: np.ndarray) -> np.ndarray:
    """Dataset SO(3) log, partitioned into ordinary and half-turn rotations."""
    rotations = np.asarray(rotms, dtype=np.float32)
    cosine = (np.einsum("nii->n", rotations) - 1.0) / 2.0
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    half_turn = np.abs(angles - np.pi) <= 1e-6
    regular = (angles > 1e-6) & ~half_turn
    vectors = np.zeros((len(rotations), 3))
    if regular.any():
        selected = rotations[regular]
        antisymmetric = selected - selected.transpose(0, 2, 1)
        axes = antisymmetric[:, (2, 0, 1), (1, 2, 0)]
        axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)
        vectors[regular] = axes * angles[regular, None]
    for index in np.flatnonzero(half_turn):
        vectors[index] = _axis_from_pi(rotations[index]) * angles[index]
    return vectors


def aa2rotm(axis_angle) -> np.ndarray:
    """Rodrigues exponential using a skew operator built from basis vectors."""
    vector = np.asarray(axis_angle, dtype=np.float32)
    radians = float(np.linalg.norm(vector))
    unit = vector / (radians + 1e-10)
    identity = np.eye(3, dtype=np.float32)
    skew = np.cross(unit, identity).T
    first_order = np.sin(radians) * skew
    second_order = (1.0 - np.cos(radians)) * skew @ skew
    return identity + first_order + second_order


def validate_stats(
    mean: Sequence[Sequence[float]], std: Sequence[Sequence[float]], action_length: int
):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (action_length, ACTION_DIM):
        raise ValueError(
            f"mean expected shape {(action_length, ACTION_DIM)}, got {mean.shape}"
        )
    if std.shape != (action_length, ACTION_DIM):
        raise ValueError(
            f"std expected shape {(action_length, ACTION_DIM)}, got {std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0):
        raise ValueError(
            "action statistics must be finite with nonnegative standard deviations"
        )
    return mean, std


def validate_quantiles(q01: Sequence[Sequence[float]], q99: Sequence[Sequence[float]]):
    limits = tuple(np.asarray(bound, dtype=np.float32) for bound in (q01, q99))
    for bound in limits:
        if bound.shape != (1, STATE_DIM):
            raise ValueError(
                f"state quantiles expected shape {(1, STATE_DIM)}, got {bound.shape}"
            )
        if not np.isfinite(bound).all():
            raise ValueError("state quantiles must be finite")
    lower, upper = limits
    if np.any(upper < lower):
        raise ValueError("q99 must be greater than or equal to q01")
    return limits


def build_action_mask(action_length: int, temporal_mask=None) -> np.ndarray:
    active = (
        np.ones(action_length, dtype=np.int32)
        if temporal_mask is None
        else np.asarray(temporal_mask, dtype=np.int32)
    )
    if active.shape != (action_length,):
        raise ValueError("temporal mask must contain one entry per action step")
    # The action18 contract has no reserved columns: all channels share validity.
    return np.repeat(active[:, None], ACTION_DIM, axis=1)


def compose_action(
    left_ee_pos,
    left_ee_aa,
    left_gripper,
    right_ee_pos,
    right_ee_aa,
    right_gripper,
    waist,
    base,
    action_length: int,
) -> np.ndarray:
    values = (
        left_ee_pos,
        left_ee_aa,
        left_gripper,
        right_ee_pos,
        right_ee_aa,
        right_gripper,
        waist,
        base,
    )
    return pack_fields(dict(zip(_PART_NAMES, values)), (action_length,))


def compose_state(left_gripper, left_joint, right_gripper, right_joint,
                  body_velocity=(0., 0., 0.), waist_position=0.) -> np.ndarray:
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    for start, joints, gripper in (
        (0, left_joint, left_gripper),
        (8, right_joint, right_gripper),
    ):
        angles = np.asarray(joints, dtype=np.float32).reshape(-1)
        if angles.size > 7:
            raise ValueError("arm joint state cannot exceed 7 dimensions")
        state[0, start : start + angles.size] = angles
        state[0, start + 7] = np.asarray(gripper, dtype=np.float32).reshape(1)[0]
    state[0, 16:19] = np.asarray(body_velocity, dtype=np.float32).reshape(3)
    state[0, 19] = float(waist_position)
    if not np.isfinite(state).all():
        raise ValueError("state observations must be finite")
    return state


def normalize_action(action, mean, std) -> np.ndarray:
    return (action - mean) / (std + ACTION_EPS)


def normalize_action_parts(values, mean, std):
    return tuple(
        normalize_action(np.asarray(part), mean[:, columns], std[:, columns])
        for part, (_, columns) in zip(values, ACTION_PARTS)
    )


def normalize_quantile(value, q01, q99) -> np.ndarray:
    sample = np.asarray(value, dtype=np.float32)
    if sample.shape != q01.shape or sample.shape != q99.shape:
        raise ValueError("state and quantiles must have the same shape")
    result = np.zeros_like(sample)
    enabled = np.where(q99 > q01)
    # Epsilon is applied to the upper endpoint before subtraction, preserving
    # the established float32 rounding and zero output for unused channels.
    span = (q99[enabled] + ACTION_EPS) - q01[enabled]
    fraction = (sample[enabled] - q01[enabled]) / span
    result[enabled] = fraction
    result[enabled] = 2.0 * result[enabled] - 1.0
    np.clip(result, -1.0, 1.0, out=result)
    return result


def denormalize_action(action, mean, std):
    return action * (std + ACTION_EPS) + mean


def split_action(action) -> dict[str, np.ndarray]:
    action = validate_action(np.asarray(action, dtype=np.float32))
    return {name: action[..., slc] for name, slc in ACTION_PARTS}


def recover_action(
    action, robot_state: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    parts = split_action(action)
    targets = {}

    for side in ("left", "right"):
        rotm = np.asarray(robot_state[f"{side}_ee_rotm"], dtype=np.float32).reshape(
            3, 3
        )
        pos = np.asarray(robot_state[f"{side}_ee_pos"], dtype=np.float32).reshape(3)
        targets[f"{side}_ee_pos"] = (
            pos[None] + parts[f"{side}_ee_pos"] @ rotm.T
        ).astype(np.float32)
        targets[f"{side}_ee_rotm"] = np.stack(
            [rotm @ aa2rotm(delta) for delta in parts[f"{side}_ee_aa"]], axis=0
        ).astype(np.float32)
        targets[f"{side}_gripper_pos"] = (
            parts[f"{side}_gripper"]
            + np.asarray(robot_state[f"{side}_gripper_pos"], dtype=np.float32).reshape(1)
        ).astype(np.float32)

    if "waist_pos" in robot_state:
        waist = np.asarray(robot_state["waist_pos"], dtype=np.float32).reshape(1)
        targets["waist_pos"] = (waist[None] + parts["waist"]).astype(np.float32)
    targets["base_vel"] = parts["base"].astype(np.float32)

    return targets
