# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache

import h5py
import numpy as np
import torch

from worlddreamer_v5.utils.normalization_registry import load_statistics, binds_manifest
from worlddreamer_v5.data.observation import ObservationTools
from worlddreamer_v5.data.action_layout import ACTION_DIM, require_action_layout, require_manifest_action_layout
from worlddreamer_v5.utils.data_paths import data_path_mapping, remap_data_paths
from worlddreamer_v5.utils.normalization import (
    compose_action,
    normalize_action_parts,
    normalize_quantile,
    resize_image,
    validate_quantiles,
    validate_stats,
    rotm2aa_batch_stable,
)


ACTIVE_ACTION_DIM = 7
LEGACY_ACTION_TARGET = "next_observed_tcp_pose_and_gripper_aperture"
COMMAND_ACTION_TARGET = "commanded_tcp_pose_and_gripper_drive_v3"
PROMPT_AGENT_RE = re.compile(r"\s*Generate actions only for Agent \d+\.")


class SceneDataset(ObservationTools):
    """One synchronized RoboFactory scene with all agents as one sample."""

    def __init__(self, params):
        self.path_mapping = data_path_mapping(params)
        self.action_length = int(params.get("action_length", 30))
        self.max_agents = int(params.get("max_agents", 4))
        self.samples_per_task = int(params.get("samples_per_task", 4200))
        self.image_augmentation = bool(params.get("image_augmentation", True))
        self.action_target = params.get("action_target", LEGACY_ACTION_TARGET)
        if self.action_target not in (LEGACY_ACTION_TARGET, COMMAND_ACTION_TARGET):
            raise ValueError(f"unsupported action target: {self.action_target}")

        stats = load_statistics(params["normalize_path"], params.get("normalization_dataset"))
        require_action_layout(stats)
        self.mean, self.std = validate_stats(
            stats["mean"], stats["std"], self.action_length
        )
        self.q01, self.q99 = validate_quantiles(stats["q01"], stats["q99"])

        with open(params["manifest_path"], "r") as file:
            manifest = json.load(file)
        require_manifest_action_layout(manifest)
        for metadata in (params, stats):
            require_action_layout(metadata)
            if (
                metadata.get("action_target", LEGACY_ACTION_TARGET)
                != self.action_target
            ):
                raise ValueError(
                    "dataset/normalization action semantics do not match configuration"
                )
        if self.action_target == COMMAND_ACTION_TARGET:
            with open(params["manifest_path"], "rb") as file:
                manifest_hash = hashlib.sha256(file.read()).hexdigest()
            self.manifest_sha256 = manifest_hash
            if not binds_manifest(stats, manifest_hash):
                raise ValueError(
                    "normalization is not bound to this command dataset manifest"
                )
        self.records = remap_data_paths(manifest["records"], self.path_mapping)
        self.samples = []
        self.task_indices = defaultdict(list)
        selected_tasks = params.get("tasks", stats.get("tasks"))
        if stats.get("tasks") is not None and not set(selected_tasks).issubset(stats["tasks"]):
            raise ValueError("selected tasks are outside the normalization profile")
        for record_index, record in enumerate(self.records):
            if selected_tasks is not None and record["task"] not in selected_tasks:
                continue
            agent_count = int(record["agent_count"])
            if not 1 <= agent_count <= self.max_agents:
                raise ValueError(
                    f"agent_count={agent_count} is outside [1, {self.max_agents}]"
                )
            if len(record["annotations"]) != agent_count:
                raise ValueError(
                    f"annotation count does not match agent_count in record {record_index}"
                )
            valid_frames = int(record.get("valid_action_frames", record["frame_count"]))
            if not 0 < valid_frames <= int(record["frame_count"]):
                raise ValueError("invalid number of recorded action frames")
            if (
                self.action_target == COMMAND_ACTION_TARGET
                and valid_frames != int(record["frame_count"]) - 1
            ):
                raise ValueError(
                    "command labels require T actions and T+1 observations"
                )
            for frame_index in range(valid_frames):
                index = len(self.samples)
                self.samples.append((record_index, frame_index))
                self.task_indices[record["task"]].append(index)

        if not self.samples:
            raise ValueError("selected tasks contain no samples")
        if selected_tasks is not None and set(self.task_indices) != set(selected_tasks):
            raise ValueError("selected tasks are missing from the manifest")
        self._h5_files = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record_index, frame = self.samples[index]
        record = self.records[record_index]
        trajectories = [
            self._read_json(path, self.path_mapping) for path in record["annotations"]
        ]
        agent_count = len(trajectories)

        states = np.zeros((self.max_agents, 1, 20), dtype=np.float32)
        actions = np.zeros((self.max_agents, self.action_length, ACTION_DIM), dtype=np.float32)
        action_masks = np.zeros_like(actions, dtype=np.int32)
        agent_mask = np.zeros(self.max_agents, dtype=np.bool_)

        for agent_index, trajectory in enumerate(trajectories):
            metadata = trajectory.get("robofactory", {})
            if (
                metadata.get("action_target", LEGACY_ACTION_TARGET)
                != self.action_target
            ):
                raise ValueError("mixed action semantics in joint annotations")
            if int(trajectory["num_frames"]) != int(record["frame_count"]):
                raise ValueError(
                    f"frame count mismatch in {record['annotations'][agent_index]}"
                )
            valid_frames = int(record.get("valid_action_frames", record["frame_count"]))
            if int(metadata.get("valid_action_frames", valid_frames)) != valid_frames:
                raise ValueError("annotation/manifest valid action counts disagree")
            steps = min(self.action_length, valid_frames - frame)
            if steps <= 0:
                raise IndexError(f"invalid frame={frame} in record={record_index}")

            left = self._arm_action(trajectory, "left", frame, steps)
            zeros3 = np.zeros((self.action_length, 3), dtype=np.float32)
            zeros1 = np.zeros((self.action_length, 1), dtype=np.float32)
            action_parts = normalize_action_parts(
                (*left, zeros3, zeros3, zeros1, zeros1, zeros3),
                self.mean,
                self.std,
            )
            actions[agent_index] = compose_action(
                *action_parts, action_length=self.action_length
            )
            action_masks[agent_index, :steps, :ACTIVE_ACTION_DIM] = 1
            states[agent_index] = normalize_quantile(
                self._state(trajectory, frame), self.q01, self.q99
            )
            agent_mask[agent_index] = True

        base = trajectories[0]
        prompt = self._joint_prompt(base, agent_count)
        images = self._images(base, prompt["images"], frame)
        if self.image_augmentation:
            images = self._augment(images)
        else:
            images = [resize_image(image, factor=32, max_pixels=160000) for image in images]
        relative_pose = self._relative_pose(base, frame, agent_count)

        return {
            "messages": self._messages(prompt["conversations"], images),
            "action": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(action_masks),
            "state": torch.from_numpy(states),
            "agent_mask": torch.from_numpy(agent_mask),
            "relative_pose": torch.from_numpy(relative_pose),
            "task": record["task"],
            "record_index": record_index,
            "frame_index": frame,
        }

    @staticmethod
    @lru_cache(maxsize=64)
    def _read_json(path, mapping=()):
        with open(path, "r") as file:
            return remap_data_paths(json.load(file), mapping)

    def _arm_action(self, trajectory, arm, frame, steps):
        if self.action_target != COMMAND_ACTION_TARGET:
            return super()._arm_action(trajectory, arm, frame, steps)
        rotation = self._frame(trajectory, f"proprios.{arm}_ee_rotm", frame).reshape(
            3, 3
        )
        position = self._frame(trajectory, f"proprios.{arm}_ee_pos", frame)
        target_position = self._future(
            trajectory, f"actions.{arm}_ee_pos", frame, steps
        )
        target_rotation = self._future(
            trajectory, f"actions.{arm}_ee_rotm", frame, steps
        ).reshape(-1, 3, 3)
        return (
            self._pad((rotation.T @ (target_position - position).T).T, steps),
            self._pad(rotm2aa_batch_stable(rotation.T @ target_rotation), steps),
            self._delta(trajectory, f"proprios.{arm}_gripper_pos",
                        f"actions.{arm}_gripper_pos", frame, steps),
        )

    @staticmethod
    def _joint_prompt(trajectory, agent_count):
        prompt = copy.deepcopy(trajectory["instruction"]["general"][0])
        conversations = prompt["conversations"]
        human = PROMPT_AGENT_RE.sub("", conversations[0]["value"])
        human = human.replace(
            "Task instruction: Collaboratively",
            "Task instruction: All robot agents should collaboratively",
            1,
        )
        conversations[0]["value"] = human + " /no_cot"
        conversations[1]["value"] = "<cot></cot>" + "<state>" * agent_count
        return prompt

    def _relative_pose(self, trajectory, frame, agent_count):
        camera_data = trajectory.get("robofactory", {}).get("camera_data", {})
        source_h5 = camera_data.get("source_h5")
        views = camera_data.get("views", {})
        if not source_h5:
            raise ValueError("robofactory.camera_data.source_h5 is required")
        handle = self._h5_files.get(source_h5)
        if handle is None:
            handle = h5py.File(source_h5, "r")
            self._h5_files[source_h5] = handle

        transforms = []
        for agent_index in range(agent_count):
            view = views[f"agent_{agent_index + 1}_wrist"]
            transforms.append(
                np.asarray(handle[view["cam2world_gl_h5_key"]][frame], dtype=np.float32)
            )

        features = np.zeros((self.max_agents, self.max_agents, 10), dtype=np.float32)
        for source in range(agent_count):
            source_rot = transforms[source][:3, :3]
            source_pos = transforms[source][:3, 3]
            for target in range(agent_count):
                target_rot = transforms[target][:3, :3]
                target_pos = transforms[target][:3, 3]
                translation = source_rot.T @ (target_pos - source_pos)
                rotation = source_rot.T @ target_rot
                rotation_6d = rotation[:, :2].T.reshape(-1)
                features[source, target] = np.concatenate(
                    [translation, rotation_6d, [np.linalg.norm(translation)]],
                )
        return features

    def __del__(self):
        for handle in getattr(self, "_h5_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass
