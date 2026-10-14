# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

LEGACY_ACTION_CONTRACT = 'next_observed_tcp_pose_and_gripper_aperture'
COMMAND_ACTION_CONTRACT = 'commanded_tcp_pose_and_gripper_drive_v3'
# Installed Panda PDJointPosMimicController limits, per finger, unchanged.
FINGER_DRIVE_LOWER = -.01
FINGER_DRIVE_UPPER = .04


def require_action_contract(expected, *metadata):
    if expected not in (LEGACY_ACTION_CONTRACT, COMMAND_ACTION_CONTRACT):
        raise ValueError(f'unsupported action contract: {expected}')
    for item in metadata:
        actual = item.get('action_target', LEGACY_ACTION_CONTRACT)
        if actual != expected:
            raise ValueError(f'action contract mismatch: expected {expected}, found {actual}')
    return expected


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_action_contract(checkpoint, normalize):
    """Fail closed for new statistics with an unmarked or different checkpoint.

    A new training run must save a bound sidecar next to its completed checkpoint.
    Existing unmarked weights remain legacy; a CLI flag cannot relabel them.
    """
    checkpoint, normalize = Path(checkpoint), Path(normalize)
    stats = json.loads(normalize.read_text())
    sidecar = checkpoint.with_name(checkpoint.name+'.action_contract.json')
    if not sidecar.exists():
        return require_action_contract(LEGACY_ACTION_CONTRACT, stats)
    metadata = json.loads(sidecar.read_text())
    contract = require_action_contract(metadata['action_target'], stats)
    if metadata.get('checkpoint_sha256') != file_sha256(checkpoint):
        raise ValueError('checkpoint action-contract sidecar does not match weights')
    if metadata.get('normalize_sha256') != file_sha256(normalize):
        raise ValueError('checkpoint action-contract sidecar does not match normalization')
    return contract


def command_labels_to_observation_length(command_values, observation_count):
    """Command t belongs to observation t. Only append terminal hold padding."""
    values = np.asarray(command_values)
    if len(values) == 0 or len(values)+1 != observation_count or not np.isfinite(values).all():
        raise ValueError('need finite T commands and T+1 observations')
    return np.concatenate([values, values[-1:]], axis=0)


def source_legacy_numpy_seed(source_h5, env_seed):
    """Reproduce a saved demonstration, not the evaluation seed-selection rule.

    Old replay manifests contain a separate NumPy compatibility seed. The episode
    seed alone does not reconstruct upstream global-NumPy object randomization.
    """
    manifest = json.loads((Path(source_h5).parent / 'validation.json').read_text())
    recorded_env = manifest.get('episode_seed', manifest.get('seed'))
    if recorded_env is None or int(recorded_env) != env_seed:
        raise ValueError('source manifest environment seed mismatch')
    if 'legacy_numpy_seed' in manifest:
        return int(manifest['legacy_numpy_seed'])
    if manifest.get('expert_version') == 'staged_geometry_v2':
        return int(recorded_env)
    raise ValueError('source manifest lacks a reproducible legacy NumPy seed')


def drive_aperture_from_command(command):
    command=np.asarray(command,dtype=np.float64)
    if not np.isfinite(command).all() or np.any(np.abs(command)>1.00001):
        raise ValueError('gripper command must be finite in [-1,1]')
    return 2*FINGER_DRIVE_LOWER+(np.clip(command,-1,1)+1)*(FINGER_DRIVE_UPPER-FINGER_DRIVE_LOWER)


def command_from_drive_aperture(aperture):
    aperture=np.asarray(aperture,dtype=np.float64)
    if not np.isfinite(aperture).all():
        raise ValueError('non-finite gripper drive aperture')
    return np.clip((aperture-2*FINGER_DRIVE_LOWER)/(FINGER_DRIVE_UPPER-FINGER_DRIVE_LOWER)-1,-1,1)


def calibration_for_task(path, task, agent_count):
    payload=json.loads(Path(path).read_text())
    version=payload.get('schema_version',1)
    if version==1:
        transforms=np.broadcast_to(np.asarray(payload['tcp_to_camera_gl'],dtype=np.float64),(agent_count,4,4)).copy()
    elif version==2:
        entries=payload['tasks'][task]
        transforms=np.stack([np.asarray(entries[f'wrist_camera_agent{i}']['tcp_to_camera_gl'],dtype=np.float64)
                             for i in range(agent_count)])
    else:
        raise ValueError(f'unsupported calibration version {version}')
    if transforms.shape!=(agent_count,4,4) or not np.isfinite(transforms).all():
        raise ValueError('invalid per-agent calibration shape or values')
    for transform in transforms:
        rotation=transform[:3,:3]
        if not np.allclose(transform[3],[0,0,0,1],atol=1e-6) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rotation),1,atol=1e-4):
            raise ValueError('calibration is not a rigid transform')
    return transforms


def tcp_from_cameras(world_cameras, calibrations):
    cameras=np.asarray(world_cameras,dtype=np.float64)
    transforms=np.asarray(calibrations,dtype=np.float64)
    if cameras.ndim!=3 or cameras.shape[1:]!=(4,4):
        raise ValueError('expected camera transforms [A,4,4]')
    if transforms.shape==(4,4):
        transforms=np.broadcast_to(transforms,cameras.shape)
    if transforms.shape!=cameras.shape or not np.isfinite(cameras).all() or not np.isfinite(transforms).all():
        raise ValueError('camera and calibration shapes/values do not match')
    return cameras@np.linalg.inv(transforms)
