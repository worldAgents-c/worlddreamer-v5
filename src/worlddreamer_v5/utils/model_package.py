# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import json
from pathlib import Path

from .checkpoint import file_sha256

CONTRACT_FIELDS = (
    "state_dim", "state_history_length", "action_layout", "state_layout", "normalization_dataset", "action_target", "action_dim", "action_length", "normalize_sha256",
    "video_observed_latents", "video_future_latents", "sample_rate_hz",
    "action_active_dimensions", "state_active_dimensions",
)


def package_model_config(directory):
    """Resolve architecture assets relative to a relocated model directory."""
    from worlddreamer_v5.models.variants import normalize_model_config

    directory = Path(directory).resolve()
    config = json.loads((directory / "model_config.json").read_text())["model"]
    for key in ("vlm_config_path", "video_config_path"):
        relative = Path(config[key])
        target = (directory / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(directory):
            raise ValueError(f"model asset must stay inside its package: {key}")
        if not target.exists():
            raise FileNotFoundError(target)
        config[key] = str(target)
    return normalize_model_config(config)


def verify_package_assets(directory, metadata):
    directory = Path(directory).resolve()
    for name, expected in metadata["asset_sha256"].items():
        target = (directory / name).resolve()
        if Path(name).is_absolute() or not target.is_relative_to(directory):
            raise ValueError("asset path escapes model package")
        if file_sha256(target) != expected:
            raise ValueError(f"model asset checksum mismatch: {name}")


def serving_contract(config, metadata):
    """Validate inference semantics without opening a training manifest."""
    from worlddreamer_v5.data.action_layout import require_action_layout
    from worlddreamer_v5.utils.contract import COMMAND_TARGET

    contract = metadata["robot_action_contract"]
    data = config["data"]
    require_action_layout(contract)
    from worlddreamer_v5.models.control_profiles import validate_control_metadata
    validate_control_metadata(contract)
    if contract.get("state_dim") != 20:
        raise ValueError("inference requires native state20 metadata")
    if not contract.get("normalization_dataset"):
        raise ValueError("inference requires a normalization_dataset")
    if contract["action_target"] != COMMAND_TARGET:
        raise ValueError("this server requires the commanded-action adapter")
    for key in ("state_dim", "normalization_dataset", "action_dim", "action_target", "action_length", "video_observed_latents", "video_future_latents"):
        if data.get(key) != contract.get(key):
            raise ValueError(f"inference checkpoint/data contract mismatch: {key}")
    if contract["action_length"] != 30:
        raise ValueError("server requires 30 action steps")
    if file_sha256(data["normalize_path"]) != contract["normalize_sha256"]:
        raise ValueError("inference normalization checksum mismatch")
    conditioning = dict(config["conditioning"])
    if file_sha256(conditioning["prompt_cache"]) != conditioning["prompt_cache_sha256"]:
        raise ValueError("inference conditioning checksum mismatch")
    if not conditioning.get("prompt_keys"):
        raise ValueError("inference requires an explicit task conditioning map")
    return contract, conditioning
