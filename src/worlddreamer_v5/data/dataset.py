# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import json
from functools import lru_cache

import torch
from safetensors.torch import load_file

from worlddreamer_v5.data.scene_dataset import SceneDataset, COMMAND_ACTION_TARGET
from worlddreamer_v5.utils.data_paths import remap_data_paths


class WorldDataset(SceneDataset):
    """Joint action samples augmented with cached causal video inputs."""

    def __init__(self, params):
        super().__init__(params)
        self.max_views = self.max_agents + 1
        self.observed_latents = int(params.get("video_observed_latents", 3))
        self.future_latents = int(params.get("video_future_latents", 2))
        self.temporal_compression = int(params.get("video_temporal_compression", 4))
        with open(params["video_cache_manifest"], "r") as file:
            cache = json.load(file)
        if (
            self.action_target == COMMAND_ACTION_TARGET
            and cache.get("source_manifest_sha256") != self.manifest_sha256
        ):
            raise ValueError(
                "Video cache is not bound to this repaired-observation manifest; regenerate it"
            )
        if cache["record_count"] != len(self.records):
            raise ValueError(
                "Video cache and action manifest record counts do not match"
            )
        if int(cache["temporal_compression"]) != self.temporal_compression:
            raise ValueError(
                "Video cache temporal compression does not match dataset config"
            )
        cache = remap_data_paths(cache, self.path_mapping)
        self.cache_records = cache["records"]
        self.prompt_keys = cache["prompt_keys"]
        self.prompt_tensors = load_file(cache["prompt_cache"], device="cpu")

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_cache(path):
        return load_file(path, device="cpu")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        record_index = int(item["record_index"])
        frame = int(item["frame_index"])
        cache_info = self.cache_records[str(record_index)]
        cache = self._read_cache(cache_info["path"])
        latents = cache["latents"]
        view_count, channels, latent_count, latent_height, latent_width = latents.shape
        if view_count != int(self.records[record_index]["agent_count"]) + 1:
            raise ValueError(f"cached view count mismatch for record {record_index}")

        current = frame // self.temporal_compression
        offsets = torch.arange(-self.observed_latents + 1, self.future_latents + 1)
        indices = current + offsets
        latent_valid = (indices >= 0) & (indices < latent_count)
        indices = indices.clamp(0, latent_count - 1)
        observed_indices = indices[: self.observed_latents]

        video_latents = torch.zeros(
            self.max_views,
            channels,
            len(indices),
            latent_height,
            latent_width,
            dtype=latents.dtype,
        )
        video_latents[:view_count] = latents[:, :, indices]
        view_mask = torch.zeros(self.max_views, dtype=torch.bool)
        view_mask[:view_count] = True

        intrinsic = (
            torch.eye(3)
            .view(1, 1, 3, 3)
            .repeat(self.max_views, self.observed_latents, 1, 1)
        )
        extrinsic = torch.zeros(self.max_views, self.observed_latents, 3, 4)
        extrinsic[..., :3] = torch.eye(3)
        depth = torch.zeros(
            self.max_views, self.observed_latents, 15, 20, dtype=cache["depth"].dtype
        )
        depth_valid = torch.zeros(
            self.max_views, self.observed_latents, 15, 20, dtype=torch.bool
        )
        intrinsic[:view_count] = cache["intrinsics"][:, observed_indices]
        extrinsic[:view_count] = cache["extrinsics"][:, observed_indices]
        depth[:view_count] = cache["depth"][:, observed_indices]
        depth_valid[:view_count] = cache["depth_valid"][:, observed_indices].bool()

        camera_roles = torch.zeros(self.max_views, dtype=torch.long)
        camera_roles[1:view_count] = 1
        prompt_key = self.prompt_keys[item["task"]]
        item.update(
            {
                "video_latents": video_latents,
                "video_text": self.prompt_tensors[f"{prompt_key}.embeddings"],
                "video_text_mask": self.prompt_tensors[f"{prompt_key}.mask"].bool(),
                "video_view_mask": view_mask,
                "video_latent_valid": latent_valid,
                "camera_intrinsics": intrinsic,
                "camera_extrinsics": extrinsic,
                "camera_depth": depth,
                "camera_depth_valid": depth_valid,
                "camera_roles": camera_roles,
            },
        )
        return item
