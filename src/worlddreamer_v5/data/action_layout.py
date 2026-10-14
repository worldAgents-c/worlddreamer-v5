# Copyright (C) 2026 World Agents Corporation.
from dataclasses import dataclass

import numpy as np

ACTION_DIM = 18


@dataclass(frozen=True)
class ActionField:
    name: str
    start: int
    width: int
    unit: str
    semantics: str

    @property
    def columns(self):
        return slice(self.start, self.start + self.width)


FIELDS = (
    ActionField('left_ee_pos', 0, 3, 'm', 'anchor_tcp_local_delta'),
    ActionField('left_ee_aa', 3, 3, 'rad', 'anchor_tcp_relative_axis_angle'),
    ActionField('left_gripper', 6, 1, 'state_gripper_unit', 'anchor_relative_delta'),
    ActionField('right_ee_pos', 7, 3, 'm', 'anchor_tcp_local_delta'),
    ActionField('right_ee_aa', 10, 3, 'rad', 'anchor_tcp_relative_axis_angle'),
    ActionField('right_gripper', 13, 1, 'state_gripper_unit', 'anchor_relative_delta'),
    ActionField('base', 14, 3, 'm/s,m/s,rad/s', 'body_frame_vx_vy_wz'),
    ActionField('waist', 17, 1, 'm', 'anchor_relative_delta'),
)
FIELD_BY_NAME = {field.name: field for field in FIELDS}
LEFT_ARM = tuple(range(7))
RIGHT_ARM = tuple(range(7, 14))
BASE = (14, 15, 16)
WAIST = (17,)
GRIPPERS = (6, 13)


def active_dimensions(*, left=False, right=False, base=False, waist=False):
    return tuple(i for enabled, group in ((left, LEFT_ARM), (right, RIGHT_ARM),
                 (base, BASE), (waist, WAIST)) if enabled for i in group)


def validate_action(value):
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[-1] != ACTION_DIM or not np.isfinite(array).all():
        raise ValueError('action18 requires finite values with final dimension 18')
    return array


def pack_fields(values, leading_shape, *, dtype=np.float32):
    unknown = set(values) - set(FIELD_BY_NAME)
    if unknown:
        raise ValueError(f'unknown action fields: {sorted(unknown)}')
    result = np.zeros((*leading_shape, ACTION_DIM), dtype=dtype)
    for name, value in values.items():
        result[..., FIELD_BY_NAME[name].columns] = value
    return validate_action(result)


def require_action_layout(metadata):
    if metadata.get('action_dim') != ACTION_DIM:
        raise ValueError(f'expected explicit {ACTION_DIM}; convert assets offline before loading')


def require_manifest_action_layout(metadata):
    if 'action_dim' not in metadata and metadata.get('action_schema') in (
        'worlddreamer-v5.action18/v2', 'wmd.action18/v2'
    ):
        return
    require_action_layout(metadata)
