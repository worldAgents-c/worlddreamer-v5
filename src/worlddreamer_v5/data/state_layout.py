# Copyright (C) 2026 World Agents Corporation.
import numpy as np

STATE_DIM = 20
LEFT_JOINTS = slice(0, 7)
LEFT_GRIPPER = 7
RIGHT_JOINTS = slice(8, 15)
RIGHT_GRIPPER = 15
BASE = slice(16, 19)
WAIST = 19


def validate_state(value):
    state = np.asarray(value, dtype=np.float32)
    if state.ndim < 1 or state.shape[-1] != STATE_DIM or not np.isfinite(state).all():
        raise ValueError("expected finite state with 20 channels")
    return state
