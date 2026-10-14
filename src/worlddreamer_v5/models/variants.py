# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MODEL_NAME = "worlddreamer-v5"
MODEL_VARIANT = "lite"
MODEL_VARIANTS = ("lite", "stand", "large")


def get_model_profile(variant: str = MODEL_VARIANT) -> dict[str, Any]:
    """Return a fresh complete architecture specification for the selected model."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unsupported whole-model variant: {variant!r}")
    profile_path = Path(__file__).parent / "profiles" / f"{variant}.json"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"whole-model variant {variant!r} requires JSON configuration: {profile_path}"
        )
    profile = json.loads(profile_path.read_text())
    action, text = profile["action"], profile["vlm"]["text_config"]
    if action["layers"] > text["num_hidden_layers"]:
        raise ValueError("action depth exceeds the available language cache")
    if (action["head_dim"] != text["head_dim"]
            or action["kv_heads"] != text["num_key_value_heads"]
            or action["hidden_size"] % (action["head_dim"] * action["kv_heads"])):
        raise ValueError("action and language attention layouts are incompatible")
    return profile


def require_config_match(actual: Mapping, expected: Mapping, *, label: str) -> None:
    """Check all profile fields; ignore serialization metadata and extra defaults."""
    for key, value in expected.items():
        field = f"{label}.{key}"
        if key not in actual:
            raise ValueError(f"{field} is missing from the asset configuration")
        candidate = actual[key]
        if isinstance(value, dict):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"{field} must be a mapping")
            require_config_match(candidate, value, label=field)
        elif candidate != value:
            raise ValueError(f"{field}: expected {value!r}, got {candidate!r}")


def validate_asset_config(path, expected, *, component):
    """Validate assets without allowing them to redefine the model architecture."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    actual = json.loads(path.read_text())
    if component == "video" and "num_decoder_layers" in actual:
        previous = actual.pop("num_decoder_layers")
        if "num_generation_layers" in actual and actual["num_generation_layers"] != previous:
            raise ValueError("conflicting video generation layer counts")
        actual["num_generation_layers"] = previous
    require_config_match(actual, expected, label=component)


def normalize_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Select the whole-model profile and retain runtime and asset options."""
    values = dict(config)
    if values.get("model_name", MODEL_NAME) not in (MODEL_NAME, "world_dreamer_v5"):
        raise ValueError(f"expected model_name={MODEL_NAME!r}")
    variant = values.pop("model_variant", values.get("variant", MODEL_VARIANT))
    if "variant" in values and values["variant"] != variant:
        raise ValueError("conflicting model variant selectors")
    profile = get_model_profile(variant)
    if "action_layers" in values:
        raise ValueError("action_layers is owned by the whole-model variant; remove it")
    for key, expected in profile["geometry"].items():
        if key in values and values[key] != expected:
            raise ValueError(f"{variant} owns {key}={expected!r}; got {values[key]!r}")
    values.update(model_name=MODEL_NAME, variant=variant)
    return values
