# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
import json
import os
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    text = os.path.expandvars(path.read_text())
    if "${" in text:
        raise ValueError("configuration contains unresolved environment variables")
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def build_model(config):
    from worlddreamer_v5.models.variants import normalize_model_config
    from worlddreamer_v5.models.worldDreamerv5 import WorldDreamerv5

    return WorldDreamerv5(**normalize_model_config(config))
