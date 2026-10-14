# Copyright (C) 2026 World Agents Corporation.
DIMENSION_KEYS = ('state_dim', 'action_dim', 'state_history_length', 'action_length')


def control_profile(state_dim=20, action_dim=18, state_history_length=1, action_length=30):
    values = (state_dim, action_dim, state_history_length, action_length)
    if any(type(value) is not int for value in values):
        raise ValueError('control profile dimensions must be integers')
    if values == (20, 18, 1, 30):
        suffix, gap = False, 10
    elif values == (14, 18, 4, 16):
        suffix, gap = True, 0
    else:
        raise ValueError(f'unsupported control profile dimensions: {values}')
    return dict(state_steps=state_history_length, state_dimensions=state_dim,
                horizon=action_length, dimensions=action_dim, query_suffix=suffix, position_gap=gap)


def control_metadata(state_dim=20, action_dim=18, state_history_length=1, action_length=30):
    control_profile(state_dim, action_dim, state_history_length, action_length)
    metadata = dict(zip(DIMENSION_KEYS, (state_dim, action_dim, state_history_length, action_length)))
    metadata['action_layout'] = [
        ['left_ee_pos', 3], ['left_ee_axis_angle', 3], ['left_gripper', 1],
        ['right_ee_pos', 3], ['right_ee_axis_angle', 3], ['right_gripper', 1],
        ['base_velocity', 3], ['waist', 1]]
    metadata['waist_unit'] = 'm'
    if state_dim == 20:
        metadata['state_layout'] = [['left_joints', 7], ['left_gripper', 1],
            ['right_joints', 7], ['right_gripper', 1], ['base_velocity', 3], ['waist', 1]]
    else:
        metadata['state_layout'] = [['ee_position', 3], ['ee_axis_angle', 3],
            ['gripper_qpos', 2], ['base_position', 3], ['base_axis_angle', 3]]
    return metadata


def validate_control_metadata(metadata, *, expected=None):
    values = [metadata.get(key) for key in DIMENSION_KEYS]
    canonical = control_metadata(*values)
    for key, value in canonical.items():
        if metadata.get(key) != value:
            raise ValueError(f'control profile mismatch: {key}')
    if expected is not None:
        for key in DIMENSION_KEYS:
            if canonical[key] != expected[key]:
                raise ValueError(f'control profile differs from model: {key}')
    return canonical
