# Copyright (C) 2026 World Agents Corporation.
import json
from pathlib import Path

from worlddreamer_v5.data.action_layout import require_action_layout
from worlddreamer_v5.data.state_layout import STATE_DIM
from worlddreamer_v5.utils.normalization import validate_quantiles, validate_stats

REGISTRY_FORMAT = "worlddreamer-v5.normalization/v2"
READABLE_REGISTRY_FORMATS = (REGISTRY_FORMAT, "wmd.normalization/v2")


def select_statistics(document, dataset=None):
    if document.get("format") in READABLE_REGISTRY_FORMATS:
        profiles = document["datasets"]
        if dataset not in profiles:
            raise ValueError(f"normalization_dataset must name one of {sorted(profiles)}")
        profile = profiles[dataset]
        if profile.get("state_dim") != STATE_DIM:
            raise ValueError("normalization state schema mismatch")
    else:
        if "datasets" in document:
            raise ValueError("unsupported normalization registry format")
        profile = document
    missing = {"mean", "std", "q01", "q99"} - set(profile)
    if missing:
        raise ValueError(f"normalization profile missing fields: {sorted(missing)}")
    require_action_layout(profile)
    validate_stats(profile["mean"], profile["std"], 30)
    validate_quantiles(profile["q01"], profile["q99"])
    return profile


def load_statistics(path, dataset=None):
    return select_statistics(json.loads(Path(path).read_text()), dataset)


def binds_manifest(profile, digest):
    return digest == profile.get("source_manifest_sha256") or digest in profile.get("source_manifest_sha256s", [])
