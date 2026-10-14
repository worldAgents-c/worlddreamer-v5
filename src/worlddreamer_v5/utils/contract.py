# Copyright (C) 2026 World Agents Corporation.
import hashlib
import json
from pathlib import Path
from worlddreamer_v5.utils.normalization_registry import load_statistics, binds_manifest
from worlddreamer_v5.data.action_layout import ACTION_DIM, require_action_layout, require_manifest_action_layout
COMMAND_TARGET = 'commanded_tcp_pose_and_gripper_drive_v3'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def data_contract(params):
    manifest = json.loads(Path(params["manifest_path"]).read_text())
    stats = load_statistics(params["normalize_path"], params.get("normalization_dataset"))
    require_manifest_action_layout(manifest)
    for item in (params, stats):
        require_action_layout(item)
    manifest_hash = sha256(params["manifest_path"])
    target = params.get('action_target')
    if target != COMMAND_TARGET or any(
        item.get("action_target") != target
        for item in (params, manifest, stats)
    ):
        raise ValueError("training needs consistent supported action semantics")
    if not binds_manifest(stats, manifest_hash):
        raise ValueError("normalization is not bound to this dataset")
    contract = {
        "state_dim": 20,
        "normalization_dataset": params.get("normalization_dataset"),
        "action_target": target,
        "action_dim": ACTION_DIM,
        "manifest_sha256": manifest_hash,
        "normalize_sha256": sha256(params["normalize_path"]),
        "sampler_seed": int(params.get("seed", 42)),
        "samples_per_task": int(params.get("samples_per_task", 4200)),
        "action_length": int(params.get("action_length", 30)),
    }
    from worlddreamer_v5.models.control_profiles import control_metadata
    contract.update(control_metadata(action_length=contract["action_length"]))
    if cache_path := params.get("video_cache_manifest"):
        cache = json.loads(Path(cache_path).read_text())
        if cache.get("source_manifest_sha256") != manifest_hash:
            raise ValueError("Video cache is not bound to this dataset")
        count = len(manifest["records"])
        if cache.get("record_count") != count or set(cache.get("records", {})) != {
            str(i) for i in range(count)
        }:
            raise ValueError("Video cache records are incomplete")
        if cache.get("temporal_compression") != int(
            params.get("video_temporal_compression", 4)
        ):
            raise ValueError("Video cache temporal compression mismatch")
        contract.update(
            video_cache_manifest_sha256=sha256(cache_path),
            video_observed_latents=int(params.get("video_observed_latents", 3)),
            video_future_latents=int(params.get("video_future_latents", 2)),
        )
    return contract
