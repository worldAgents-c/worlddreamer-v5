# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ik_controller import IKExecutionConfig, RobosuiteStyleIKExecutor, _as_pose


@dataclass(frozen=True)
class LimitAwareConfig:
    position_tolerance_m: float = 0.0002
    rotation_tolerance_rad: float = 0.002
    limit_margin_rad: float = 0.20
    max_joint_jump_rad: float = 0.25
    max_evaluations: int = 60
    posture_weight: float = 0.02
    continuity_weight: float = 0.05
    limit_weight: float = 0.10

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


class LimitAwarePlanner:
    """A narrow Planner adapter: replace IK, preserve FK and collision semantics."""

    def __init__(self, planner, config=LimitAwareConfig()):
        self.base = planner
        self.config = config
        self.indices = np.asarray(planner.move_group_joint_indices, dtype=int)
        self.limits = np.asarray(planner.joint_limits, dtype=float)[self.indices]
        if self.limits.shape != (len(self.indices), 2) or not np.isfinite(self.limits).all():
            raise ValueError('finite joint limits required')
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError('empty joint range')
        self.anchor = None
        self.last_diagnostics = {}
        self.attempt_diagnostics = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _pose(self, qpos):
        model = self.base.pinocchio_model
        model.compute_forward_kinematics(qpos)
        return _as_pose(np.asarray(model.get_link_pose(self.base.link_name_2_idx[self.base.move_group])))

    @staticmethod
    def _rotation(pose):
        return Rotation.from_quat(pose[[4, 5, 6, 3]])

    def IK(self, goal_pose, start_qpos, mask=None, n_init_qpos=1, threshold=1e-3):
        if mask is not None and len(mask):
            raise ValueError('extra joint masks are not supported by this executor')
        if n_init_qpos != 1:
            raise ValueError('limit-aware IK is deterministic; use exactly one initial configuration')
        started = time.monotonic()
        cfg = self.config
        current = np.asarray(start_qpos, dtype=float).reshape(-1)
        goal = _as_pose(goal_pose)
        if not np.isfinite(current).all():
            raise ValueError('nonfinite qpos')
        arm = current[self.indices]
        lower, upper = self.limits.T
        if self.anchor is None:
            self.anchor = np.clip(arm, lower, upper).copy()
        # This is a joint-space trust region, not action interpolation. A target
        # that cannot be reached safely is explicitly rejected/backed off.
        lo = np.maximum(lower, arm - cfg.max_joint_jump_rad)
        hi = np.minimum(upper, arm + cfg.max_joint_jump_rad)
        if np.any(lo >= hi):
            return 'IK Failed! Current arm is outside the bounded trust region.', []
        target_rotation = self._rotation(goal)
        span = upper - lower
        margin = np.minimum(cfg.limit_margin_rad, span * .2)
        candidate = current.copy()

        def residual(q):
            candidate[self.indices] = q
            pose = self._pose(candidate)
            rotation_error = (target_rotation * self._rotation(pose).inv()).as_rotvec()
            # Task residuals dominate posture costs. Acceptance below separately
            # checks geometric error, limits, and the existing collision filter.
            return np.r_[
                (pose[:3] - goal[:3]) / .005,
                rotation_error / .05,
                cfg.posture_weight * (q - self.anchor) / span,
                cfg.continuity_weight * (q - arm),
                cfg.limit_weight * np.maximum(0, margin - (q - lower)) / margin,
                cfg.limit_weight * np.maximum(0, margin - (upper - q)) / margin,
            ]

        result = least_squares(residual, np.clip(arm, lo+1e-10, hi-1e-10),
            bounds=(lo, hi), method='trf', max_nfev=cfg.max_evaluations,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        candidate[self.indices] = result.x
        pose = self._pose(candidate)
        position_error = float(np.linalg.norm(pose[:3] - goal[:3]))
        rotation_error = float(np.linalg.norm((target_rotation * self._rotation(pose).inv()).as_rotvec()))
        valid = bool(np.isfinite(candidate).all()
                     and position_error <= min(cfg.position_tolerance_m, threshold)
                     and rotation_error <= cfg.rotation_tolerance_rad)
        collisions = -1
        limits_valid = False
        if valid:
            limits_valid = bool(self.base.wrap_joint_limit(candidate))
            valid = limits_valid
        if valid:
            self.base.robot.set_qpos(current, True)
            self.base.planning_world.set_qpos_all(candidate[self.indices])
            collisions = len(self.base.planning_world.collide_full())
            valid = collisions == 0
        self.last_diagnostics = {
            'position_error_m': position_error, 'rotation_error_rad': rotation_error,
            'evaluations': int(result.nfev), 'collisions': collisions,
            'limits_valid':limits_valid,'candidate_qpos':candidate.tolist(),
            'joint_margin_rad': float(np.min(np.minimum(result.x-lower, upper-result.x))),
            'latency_seconds': time.monotonic()-started,
        }
        self.attempt_diagnostics.append(self.last_diagnostics)
        if not valid:
            return (f'IK Failed! bounded solve pos={position_error:.6g}, '
                    f'rot={rotation_error:.6g}, collisions={collisions}'), []
        return 'Success', [candidate.copy()]


class LimitAwareIKExecutor(RobosuiteStyleIKExecutor):
    def __init__(self, planner, config=IKExecutionConfig(rate_limit_steps=False),
                 limit_config=LimitAwareConfig()):
        if config.rate_limit_steps:
            raise ValueError('limit-aware mode preserves EE targets; disable the old rate limiter')
        super().__init__(LimitAwarePlanner(planner, limit_config), config)

    def solve(self, desired_pose, current_qpos):
        self.planner.attempt_diagnostics.clear()
        return super().solve(desired_pose,current_qpos)
