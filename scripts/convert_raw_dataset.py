# Copyright (C) 2026 World Agents Corporation.
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


TASKS = {"PlaceFood", "CameraAlignment", "TakePhoto"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve(root, name, prefix):
    if not name.startswith(prefix + "/"):
        raise ValueError(f"Expected {prefix}/ relative path")
    relative = Path(name[len(prefix) + 1:])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset path escapes its root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Dataset file is missing or outside its root")
    return path, relative


def convert(raw, cache, output):
    source = read(raw / "manifest.json")
    if source.get("format") != "worlddreamer-v5.robofactory-raw/v1":
        raise ValueError("Unsupported raw dataset format")
    records = source["records"]
    if {r["task"] for r in records} != TASKS or len(records) != source["record_count"]:
        raise ValueError("Dataset task selection or record count differs")
    if len({(r["task"], r["episode"]) for r in records}) != len(records):
        raise ValueError("Duplicate episode")
    statistics = read(raw / "normalize.json")
    profile = statistics["datasets"]["RoboFactory"]
    if profile["source_manifest_sha256"] != digest(raw / "manifest.json"):
        raise ValueError("Normalization does not match the raw manifest")
    if (profile["state_dim"], profile["action_dim"]) != (20, 18):
        raise ValueError("Expected state20 and action18")
    video = read(cache / "manifest.json")
    if video["source_manifest_sha256"] != source["prepared_manifest_sha256"]:
        raise ValueError("Video cache does not match this dataset")
    if set(video["records"]) != {str(i) for i in range(len(records))}:
        raise ValueError("Incomplete video cache")
    files, prepared, labels = {}, [], {}

    def collect(root, path, prefix, destination_prefix=""):
        full, relative = resolve(root, path, prefix)
        destination = Path(destination_prefix) / relative
        if str(destination) in files and files[str(destination)] != full:
            raise ValueError("Conflicting output paths")
        files[str(destination)] = full

    for index, record in enumerate(records):
        item = copy.deepcopy(record)
        label_path, _ = resolve(raw, item.pop("labels"), "@data")
        agents = read(label_path)["agents"]
        if len(agents) != record["agent_count"]:
            raise ValueError("Agent count differs from labels")
        names = []
        for agent, annotation in enumerate(agents, 1):
            if annotation["num_frames"] != record["frame_count"]:
                raise ValueError("Label frame count differs")
            metadata = annotation["robofactory"]
            if metadata["active_source_agent"] != agent or metadata["action_target"] != source["action_target"]:
                raise ValueError("Agent order or action semantics differ")
            name = f"episodes/{record['task']}/{record['episode']:06d}/agent_{agent}.json"
            labels[name] = annotation
            names.append("@data/" + name)
        item["annotations"] = names
        prepared.append(item)
        for path in [record["source_h5"], record["config"], *record["videos"].values()]:
            collect(raw, path, "@data")
        entry = video["records"][str(index)]
        if (entry["task"], entry["episode"]) != (record["task"], record["episode"]):
            raise ValueError("Video cache episode order differs")
        collect(cache, entry["path"], "@cache", "cache")
    collect(cache, video["prompt_cache"], "@cache", "cache")
    manifest = {"state_dim": 20, "action_dim": 18,
                "action_target": source["action_target"], "tasks": source["tasks"],
                "record_count": len(prepared), "records": prepared}
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    target_digest = hashlib.sha256(payload).hexdigest()
    if target_digest != source["prepared_manifest_sha256"]:
        raise ValueError("Converted manifest differs from the paired dataset")
    output.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name, annotation in labels.items():
        write(output / name, annotation)
    shutil.copyfile(raw / "calibration.json", output / "calibration.json")
    (output / "manifest.json").write_bytes(payload)
    profile["source_manifest_sha256"] = target_digest
    write(output / "normalize.json", statistics)
    write(output / "cache/manifest.json", video)
    return {"records": len(prepared), "annotations": len(labels),
            "state_dim": 20, "action_dim": 18, "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.raw, args.cache, args.output)))
