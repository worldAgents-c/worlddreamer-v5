# Copyright (C) 2026 World Agents Corporation.
import hashlib
import json
from pathlib import Path

FORMAT_VERSION = 2


def read_video_config(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'config.json'
    config = json.loads(path.read_text())
    return {key: value for key, value in config.items() if not key.startswith('_')}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state(checkpoint):
    """Recognize full ZeRO-2 model states, Lightning states, and raw tensors."""
    state = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint contains no state dictionary")
    wrapped = [key.startswith("model.") for key in state]
    if any(wrapped) and not all(wrapped):
        raise ValueError(
            "mixed model and non-model keys; select a complete model state"
        )
    if all(wrapped):
        state = {name[6:]: value for name, value in state.items()}
    return state


def load_tensor_source(path):
    import torch

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        if (
            path.name == "model.safetensors"
            and (path.parent / "checkpoint.json").is_file()
        ):
            return load_bundle(path.parent)
        return load_file(str(path), device="cpu"), {}
    if not path.is_file():
        raise ValueError(
            "select the full model state file, not an optimizer shard or directory"
        )
    source = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    return extract_state(source), source


def validate_state(state, expected=None, *, check_dtype=False):
    import torch

    if not state:
        raise ValueError("empty model state")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"non-tensor model value: {name}")
    if expected is not None:
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        if missing or unexpected:
            raise ValueError(
                f"state schema mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for name, value in state.items():
            if tuple(value.shape) != tuple(expected[name].shape):
                raise ValueError(
                    f"shape mismatch for {name}: {tuple(value.shape)} != {tuple(expected[name].shape)}"
                )
            if check_dtype and value.dtype != expected[name].dtype:
                raise ValueError(
                    f"dtype mismatch for {name}: {value.dtype} != {expected[name].dtype}"
                )


def load_model_weights(model, path, *, source_format="worlddreamer-v5"):
    """Load only the current architecture and explicit physical action schema."""
    if source_format != "worlddreamer-v5":
        raise ValueError(
            "convert source weights offline; runtime accepts only the current model format"
        )
    path = Path(path)
    state, metadata = load_bundle(path) if path.is_dir() else load_tensor_source(path)
    from worlddreamer_v5.models.control_profiles import validate_control_metadata
    contract = metadata.get("robot_action_contract", {})
    if not all(key in contract for key in ("state_dim", "action_dim", "state_history_length", "action_length")):
        raise ValueError("model metadata requires explicit dimensions; migrate offline before loading")
    expected = dict(state_dim=model.state_shape[1], action_dim=model.action_shape[1],
                    state_history_length=model.state_shape[0], action_length=model.action_shape[0])
    validate_control_metadata(contract, expected=expected)
    validate_state(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    return metadata


def load_bundle(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    metadata = json.loads((directory / "checkpoint.json").read_text())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported worlddreamer-v5 checkpoint format")
    from .model_package import verify_package_assets

    if "asset_sha256" in metadata or (directory / "assets").exists():
        verify_package_assets(directory, metadata)
    path = directory / "model.safetensors"
    if file_sha256(path) != metadata["model_sha256"]:
        raise ValueError("model file checksum mismatch")
    return load_file(str(path), device="cpu"), metadata
