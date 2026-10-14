# Copyright (C) 2026 World Agents Corporation.
from pathlib import PurePosixPath, Path


def data_path_mapping(params):
    roots = dict(params.get("data_path_mapping", {}))
    if params.get('manifest_path'):
        roots.setdefault('@data', str(Path(params['manifest_path']).resolve().parent))
    if params.get('video_cache_manifest'):
        roots.setdefault('@cache', str(Path(params['video_cache_manifest']).resolve().parent))
    pairs = tuple(sorted(roots.items()))
    for source, destination in pairs:
        for root in (source, destination):
            path = PurePosixPath(root)
            if (not path.is_absolute() and root not in ("@data", "@cache")) or ".." in path.parts or str(path) == "/":
                raise ValueError(f"invalid data relocation root: {root}")
    return pairs


def remap_data_paths(value, mapping):
    if isinstance(value, dict):
        return {key: remap_data_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_data_paths(item, mapping) for item in value]
    if isinstance(value, str):
        for source, destination in mapping:
            if value == source or value.startswith(source.rstrip("/") + "/"):
                relative = PurePosixPath(value).relative_to(source)
                if ".." in relative.parts:
                    raise ValueError(f"data path escapes relocation root: {value}")
                return str(PurePosixPath(destination) / relative)
    return value
