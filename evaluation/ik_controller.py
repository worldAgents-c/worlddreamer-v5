# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-12


def _as_pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"expected finite pose [x,y,z,qw,qx,qy,qz], got {pose}")
    pose = pose.copy()
    norm = float(np.linalg.norm(pose[3:]))
    if norm < _EPS:
        raise ValueError("pose quaternion has zero norm")
    pose[3:] /= norm
    return pose


def quaternion_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest geodesic angle between two wxyz quaternions."""

    first = _as_pose(np.r_[np.zeros(3), first])[3:]
    second = _as_pose(np.r_[np.zeros(3), second])[3:]
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * float(np.arccos(cosine))


def quaternion_slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation for wxyz quaternions."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    first = _as_pose(np.r_[np.zeros(3), first])[3:]
    second = _as_pose(np.r_[np.zeros(3), second])[3:]
    cosine = float(np.dot(first, second))
    if cosine < 0.0:
        second = -second
        cosine = -cosine
    cosine = float(np.clip(cosine, -1.0, 1.0))
    if cosine > 1.0 - 1e-8:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = float(np.arccos(cosine))
    scale = float(np.sin(angle))
    result = (
        np.sin((1.0 - fraction) * angle) / scale * first
        + np.sin(fraction * angle) / scale * second
    )
    return result / np.linalg.norm(result)


def interpolate_pose(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    first = _as_pose(first)
    second = _as_pose(second)
    return np.r_[
        first[:3] + fraction * (second[:3] - first[:3]),
        quaternion_slerp(first[3:], second[3:], fraction),
    ]


@dataclass(frozen=True)
class PoseStep:
    pose: np.ndarray
    requested_translation_m: float
    requested_rotation_rad: float
    translation_limited: bool
    rotation_limited: bool


def limit_pose_step(
    reference: np.ndarray,
    desired: np.ndarray,
    max_translation_m: float,
    max_rotation_rad: float,
) -> PoseStep:
    """Limit one desired absolute pose relative to the previous command pose."""

    if max_translation_m <= 0.0 or max_rotation_rad <= 0.0:
        raise ValueError("pose step limits must be positive")
    reference = _as_pose(reference)
    desired = _as_pose(desired)
    delta = desired[:3] - reference[:3]
    translation = float(np.linalg.norm(delta))
    rotation = quaternion_angle(reference[3:], desired[3:])
    translation_fraction = min(1.0, max_translation_m / max(translation, _EPS))
    rotation_fraction = min(1.0, max_rotation_rad / max(rotation, _EPS))
    pose = np.r_[
        reference[:3] + translation_fraction * delta,
        quaternion_slerp(reference[3:], desired[3:], rotation_fraction),
    ]
    return PoseStep(
        pose=pose,
        requested_translation_m=translation,
        requested_rotation_rad=rotation,
        translation_limited=translation_fraction < 1.0,
        rotation_limited=rotation_fraction < 1.0,
    )


@dataclass(frozen=True)
class IKExecutionConfig:
    max_translation_step_m: float = 0.02
    max_rotation_step_rad: float = 0.05
    max_joint_step_rad: float = 0.045
    backoff_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    ik_attempts: int = 1
    ik_threshold: float = 1e-3
    rate_limit_steps: bool = True

    def __post_init__(self) -> None:
        if self.max_translation_step_m <= 0.0:
            raise ValueError("max_translation_step_m must be positive")
        if self.max_rotation_step_rad <= 0.0:
            raise ValueError("max_rotation_step_rad must be positive")
        if self.max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive")
        if self.ik_attempts <= 0 or self.ik_threshold <= 0.0:
            raise ValueError("IK attempts and threshold must be positive")
        if not self.backoff_scales:
            raise ValueError("at least one IK backoff scale is required")
        if self.backoff_scales[0] != 1.0:
            raise ValueError("the first IK backoff scale must be 1.0")
        if any(not 0.0 < scale <= 1.0 for scale in self.backoff_scales):
            raise ValueError("IK backoff scales must be in (0, 1]")
        if any(left <= right for left, right in zip(self.backoff_scales, self.backoff_scales[1:])):
            raise ValueError("IK backoff scales must be strictly decreasing")


@dataclass(frozen=True)
class IKStepResult:
    arm_target: np.ndarray
    commanded_pose: np.ndarray
    success: bool
    solver_status: str
    solver_calls: int
    backoff_scale: float
    translation_limited: bool
    rotation_limited: bool
    joint_limited: bool
    requested_translation_m: float
    requested_rotation_rad: float
    raw_max_joint_delta_rad: float
    commanded_max_joint_delta_rad: float
    projection_position_error_m: float
    projection_rotation_error_rad: float


class RobosuiteStyleIKExecutor:
    """Warm-started deterministic IK with closest-solution selection and fallback."""

    def __init__(self, planner, config: IKExecutionConfig):
        self.planner = planner
        self.config = config
        self.indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError(f"invalid move-group joint indices: {self.indices}")
        self.reference_pose: np.ndarray | None = None

    def pose_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains non-finite values")
        model = self.planner.pinocchio_model
        model.compute_forward_kinematics(qpos)
        link_index = self.planner.link_name_2_idx[self.planner.move_group]
        return _as_pose(np.asarray(model.get_link_pose(link_index), dtype=np.float64))

    def begin_chunk(self, current_qpos: np.ndarray) -> None:
        """Synchronize the command reference once at each policy replan."""

        self.reference_pose = self.pose_for_qpos(current_qpos)

    def solve(self, desired_pose: np.ndarray, current_qpos: np.ndarray) -> IKStepResult:
        current_qpos = np.asarray(current_qpos, dtype=np.float64).reshape(-1)
        current_pose = self.pose_for_qpos(current_qpos)
        if self.reference_pose is None:
            self.reference_pose = current_pose
        desired_pose = _as_pose(desired_pose)
        if self.config.rate_limit_steps:
            pose_step = limit_pose_step(
                self.reference_pose,
                desired_pose,
                self.config.max_translation_step_m,
                self.config.max_rotation_step_rad,
            )
        else:
            pose_step = PoseStep(
                pose=desired_pose,
                requested_translation_m=float(
                    np.linalg.norm(desired_pose[:3] - self.reference_pose[:3]),
                ),
                requested_rotation_rad=quaternion_angle(
                    self.reference_pose[3:],
                    desired_pose[3:],
                ),
                translation_limited=False,
                rotation_limited=False,
            )

        last_status = "IK was not attempted"
        solver_calls = 0
        for scale in self.config.backoff_scales:
            candidate = interpolate_pose(current_pose, pose_step.pose, scale)
            status, solutions = self.planner.IK(
                candidate,
                current_qpos,
                n_init_qpos=self.config.ik_attempts,
                threshold=self.config.ik_threshold,
            )
            solver_calls += 1
            last_status = str(status)
            if status != "Success" or not solutions:
                continue

            selected = min(
                solutions,
                key=lambda value: np.linalg.norm(
                    np.asarray(value, dtype=np.float64)[self.indices] - current_qpos[self.indices],
                ),
            )
            raw_target = np.asarray(selected, dtype=np.float64)[self.indices]
            raw_delta = raw_target - current_qpos[self.indices]
            if self.config.rate_limit_steps:
                commanded_delta = np.clip(
                    raw_delta,
                    -self.config.max_joint_step_rad,
                    self.config.max_joint_step_rad,
                )
            else:
                commanded_delta = raw_delta
            unclipped_arm_target = current_qpos[self.indices] + commanded_delta
            limits = np.asarray(self.planner.joint_limits, dtype=np.float64)[self.indices]
            arm_target = np.clip(unclipped_arm_target, limits[:, 0], limits[:, 1])
            full_target = current_qpos.copy()
            full_target[self.indices] = arm_target
            commanded_pose = self.pose_for_qpos(full_target)
            self.reference_pose = commanded_pose
            return IKStepResult(
                arm_target=arm_target.astype(np.float32),
                commanded_pose=commanded_pose,
                success=True,
                solver_status=last_status,
                solver_calls=solver_calls,
                backoff_scale=float(scale),
                translation_limited=pose_step.translation_limited,
                rotation_limited=pose_step.rotation_limited,
                joint_limited=not np.allclose(raw_target, arm_target, rtol=0.0, atol=1e-12),
                requested_translation_m=pose_step.requested_translation_m,
                requested_rotation_rad=pose_step.requested_rotation_rad,
                raw_max_joint_delta_rad=float(np.max(np.abs(raw_delta))),
                commanded_max_joint_delta_rad=float(np.max(np.abs(arm_target - current_qpos[self.indices]))),
                projection_position_error_m=float(np.linalg.norm(commanded_pose[:3] - candidate[:3])),
                projection_rotation_error_rad=quaternion_angle(commanded_pose[3:], candidate[3:]),
            )

        self.reference_pose = current_pose
        arm_target = current_qpos[self.indices].astype(np.float32)
        return IKStepResult(
            arm_target=arm_target,
            commanded_pose=current_pose,
            success=False,
            solver_status=last_status,
            solver_calls=solver_calls,
            backoff_scale=0.0,
            translation_limited=pose_step.translation_limited,
            rotation_limited=pose_step.rotation_limited,
            joint_limited=False,
            requested_translation_m=pose_step.requested_translation_m,
            requested_rotation_rad=pose_step.requested_rotation_rad,
            raw_max_joint_delta_rad=0.0,
            commanded_max_joint_delta_rad=0.0,
            projection_position_error_m=float(np.linalg.norm(current_pose[:3] - pose_step.pose[:3])),
            projection_rotation_error_rad=quaternion_angle(current_pose[3:], pose_step.pose[3:]),
        )


def pose_error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first = _as_pose(first)
    second = _as_pose(second)
    return (
        float(np.linalg.norm(first[:3] - second[:3])),
        quaternion_angle(first[3:], second[3:]),
    )
