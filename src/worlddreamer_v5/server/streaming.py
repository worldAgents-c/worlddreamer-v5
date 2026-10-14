# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
from collections import deque
import numpy as np
from worlddreamer_v5.server.protocol import MAX_AGENTS, latent_seed_for, scalar_int

OBSERVED_LATENTS = 3
TEMPORAL_COMPRESSION = 4
MAX_VIEWS = MAX_AGENTS + 1


class StreamingVideoSession:
    """Incrementally reproduce the video VAE's causal 1+4+4 temporal encoding."""

    def __init__(self, vae, prompt_tensors, prompt_keys, device: str, *, layout="global_wrist"):
        import torch

        self.torch = torch
        self.vae = vae
        self.prompt_tensors = prompt_tensors
        self.prompt_keys = prompt_keys
        self.device = device
        if layout not in ("global_wrist", "front_wrist", "robocasa365"):
            raise ValueError(f"unsupported camera layout: {layout}")
        self.layout = layout
        self.max_views = (3 if layout == "robocasa365" else
                          MAX_VIEWS if layout == "global_wrist" else 2 * MAX_AGENTS)
        self.latent_mean = torch.tensor(
            vae.config.latents_mean,
            device=device,
            dtype=vae.dtype,
        ).view(1, -1, 1, 1, 1)
        self.latent_std = torch.tensor(
            vae.config.latents_std,
            device=device,
            dtype=vae.dtype,
        ).view(1, -1, 1, 1, 1)
        self.reset(0, 0, 0)

    def reset(self, env_seed: int, agent_count: int, model_seed_base: int) -> None:
        torch = self.torch
        if agent_count and not 1 <= agent_count <= MAX_AGENTS:
            raise ValueError(f"agent_count={agent_count} is outside [1, {MAX_AGENTS}]")
        self.env_seed = int(env_seed)
        self.model_seed_base = int(model_seed_base)
        self.agent_count = int(agent_count)
        if self.layout == "robocasa365" and agent_count not in (0, 1):
            raise ValueError("RoboCasa365 requires exactly one robot")
        self.view_count = ((self.agent_count + 1 if self.layout == "global_wrist"
                            else 3 if self.layout == "robocasa365"
                            else 2 * self.agent_count) if self.agent_count else 0)
        self.frame_count = 0
        self.latent_count = 0
        self.pending_rgb: list[np.ndarray] = []
        self.pending_intrinsics: list[np.ndarray] = []
        self.pending_extrinsics: list[np.ndarray] = []
        self.pending_depth: list[np.ndarray] = []
        self.pending_depth_valid: list[np.ndarray] = []
        self.pending_pixel_valid: list[np.ndarray] = []
        self.pixel_valid = [deque(maxlen=OBSERVED_LATENTS) for _ in range(self.view_count)]
        self.latents = [deque(maxlen=OBSERVED_LATENTS) for _ in range(self.view_count)]
        self.intrinsics = [
            deque(maxlen=OBSERVED_LATENTS) for _ in range(self.view_count)
        ]
        self.extrinsics = [
            deque(maxlen=OBSERVED_LATENTS) for _ in range(self.view_count)
        ]
        self.depth = [deque(maxlen=OBSERVED_LATENTS) for _ in range(self.view_count)]
        self.depth_valid = [
            deque(maxlen=OBSERVED_LATENTS) for _ in range(self.view_count)
        ]
        self.feature_caches: dict[tuple[int, int, int], list] = {}
        self.generators = [
            torch.Generator(device=self.device).manual_seed(
                latent_seed_for(self.model_seed_base, view)
            )
            for view in range(self.view_count)
        ]

    def _validate_request(self, request: dict[str, np.ndarray]) -> int:
        if not self.view_count:
            raise RuntimeError("Video session must be reset before inference")
        start = scalar_int(request["video_frame_start"])
        if start != self.frame_count:
            raise ValueError(
                f"non-contiguous video frames: expected {self.frame_count}, got {start}"
            )
        rgb = np.asarray(request["video_rgb"], dtype=np.uint8)
        if rgb.ndim != 5 or rgb.shape[0] != self.view_count or rgb.shape[-1] != 3:
            raise ValueError(f"invalid video_rgb shape: {rgb.shape}")
        frame_count = int(rgb.shape[1])
        expected = {
            "video_intrinsics": (self.view_count, frame_count, 3, 3),
            "video_extrinsics": (self.view_count, frame_count, 3, 4),
            "video_depth": (self.view_count, frame_count, 15, 20),
            "video_depth_valid": (self.view_count, frame_count, 15, 20),
        }
        for key, shape in expected.items():
            if tuple(np.asarray(request[key]).shape) != shape:
                raise ValueError(
                    f"invalid {key} shape: {np.asarray(request[key]).shape}, expected {shape}"
                )
        if "camera_pixel_valid" in request:
            pixels = np.asarray(request["camera_pixel_valid"])
            if pixels.shape != expected["video_depth_valid"] or pixels.dtype != np.bool_:
                raise ValueError("invalid camera_pixel_valid shape or dtype")
        return frame_count

    def _tile_cache(self, view: int, row: int, column: int) -> list:
        key = (view, row, column)
        if key not in self.feature_caches:
            count = int(self.vae._cached_conv_counts["encoder"])
            self.feature_caches[key] = [None] * count
        return self.feature_caches[key]

    def _encode_tile(self, video, view: int, row: int, column: int):
        cache = self._tile_cache(view, row, column)
        feature_index = [0]
        encoded = self.vae.encoder(video, feat_cache=cache, feat_idx=feature_index)
        if feature_index[0] > len(cache):
            raise RuntimeError(
                f"Video VAE encoder cache overflow: used {feature_index[0]} of {len(cache)} entries",
            )
        if hasattr(self, "encoder_cache_entries_used"):
            if feature_index[0] != self.encoder_cache_entries_used:
                raise RuntimeError(
                    "Video VAE encoder used a different number of causal cache entries between chunks",
                )
        else:
            self.encoder_cache_entries_used = feature_index[0]
        return self.vae.quant_conv(encoded)

    def _streaming_tiled_moments(self, video, view: int):
        _, _, _, height, width = video.shape
        min_height = int(self.vae.tile_sample_min_height)
        min_width = int(self.vae.tile_sample_min_width)
        stride_height = int(self.vae.tile_sample_stride_height)
        stride_width = int(self.vae.tile_sample_stride_width)
        ratio = int(self.vae.spatial_compression_ratio)
        if height <= min_height and width <= min_width:
            return self._encode_tile(video, view, 0, 0)

        tile_height = min_height // ratio
        tile_width = min_width // ratio
        latent_stride_height = stride_height // ratio
        latent_stride_width = stride_width // ratio
        blend_height = tile_height - latent_stride_height
        blend_width = tile_width - latent_stride_width
        rows = []
        for row in range(0, height, stride_height):
            values = []
            for column in range(0, width, stride_width):
                tile = video[
                    :, :, :, row : row + min_height, column : column + min_width
                ]
                values.append(self._encode_tile(tile, view, row, column))
            rows.append(values)

        result_rows = []
        for row_index, row in enumerate(rows):
            result = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self.vae.blend_v(
                        rows[row_index - 1][column_index], tile, blend_height
                    )
                if column_index > 0:
                    tile = self.vae.blend_h(row[column_index - 1], tile, blend_width)
                result.append(
                    tile[:, :, :, :latent_stride_height, :latent_stride_width],
                )
            result_rows.append(self.torch.cat(result, dim=-1))
        latent_height = height // ratio
        latent_width = width // ratio
        return self.torch.cat(result_rows, dim=3)[
            :, :, :, :latent_height, :latent_width
        ]

    def _encode_chunk(self, rgb: np.ndarray) -> list:
        torch = self.torch
        from diffusers.models.autoencoders.autoencoder_kl_wan import patchify
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        video = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 4, 1, 2, 3)
        video = video.to(device=self.device, dtype=self.vae.dtype).div_(127.5).sub_(1.0)
        if self.vae.config.patch_size is not None:
            video = patchify(video, patch_size=self.vae.config.patch_size)
        outputs = []
        for view in range(self.view_count):
            moments = self._streaming_tiled_moments(video[view : view + 1], view)
            posterior = DiagonalGaussianDistribution(moments)
            latent = posterior.sample(generator=self.generators[view])
            latent = (latent - self.latent_mean) / self.latent_std
            outputs.append(latent[0, :, 0].to(torch.bfloat16))
        return outputs

    def _consume(self, count: int) -> None:
        rgb = np.stack(self.pending_rgb[:count], axis=1)
        new_latents = self._encode_chunk(rgb)
        source = count - 1
        for view in range(self.view_count):
            self.latents[view].append(new_latents[view])
            self.intrinsics[view].append(self.pending_intrinsics[source][view])
            self.extrinsics[view].append(self.pending_extrinsics[source][view])
            self.depth[view].append(self.pending_depth[source][view])
            self.depth_valid[view].append(self.pending_depth_valid[source][view])
            self.pixel_valid[view].append(self.pending_pixel_valid[source][view])
        del self.pending_rgb[:count]
        del self.pending_intrinsics[:count]
        del self.pending_extrinsics[:count]
        del self.pending_depth[:count]
        del self.pending_depth_valid[:count]
        del self.pending_pixel_valid[:count]
        self.latent_count += 1

    def ingest(self, request: dict[str, np.ndarray]) -> int:
        frame_count = self._validate_request(request)
        rgb = np.asarray(request["video_rgb"], dtype=np.uint8)
        intrinsics = np.asarray(request["video_intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(request["video_extrinsics"], dtype=np.float32)
        depth = np.asarray(request["video_depth"], dtype=np.float16)
        depth_valid = np.asarray(request["video_depth_valid"], dtype=np.bool_)
        pixel_valid = np.asarray(request.get("camera_pixel_valid", np.ones_like(depth_valid)), dtype=np.bool_)
        for frame in range(frame_count):
            self.pending_rgb.append(rgb[:, frame])
            self.pending_intrinsics.append(intrinsics[:, frame])
            self.pending_extrinsics.append(extrinsics[:, frame])
            self.pending_depth.append(depth[:, frame])
            self.pending_depth_valid.append(depth_valid[:, frame])
            self.pending_pixel_valid.append(pixel_valid[:, frame])
        self.frame_count += frame_count

        before = self.latent_count
        if self.latent_count == 0 and self.pending_rgb:
            self._consume(1)
        while len(self.pending_rgb) >= TEMPORAL_COMPRESSION:
            self._consume(TEMPORAL_COMPRESSION)
        return self.latent_count - before

    @staticmethod
    def _observed_indices(count: int) -> tuple[list[int], list[bool]]:
        current = count - 1
        raw = [current - 2, current - 1, current]
        return [max(0, value) for value in raw], [value >= 0 for value in raw]

    def model_inputs(self, task: str) -> dict[str, object]:
        torch = self.torch
        if self.latent_count <= 0:
            raise RuntimeError("no video latent is available")
        indices, valid = self._observed_indices(self.latent_count)
        retained_start = max(0, self.latent_count - OBSERVED_LATENTS)
        local_indices = [value - retained_start for value in indices]
        sample = self.latents[0][0]
        video_latents = torch.zeros(
            1,
            self.max_views,
            sample.shape[0],
            OBSERVED_LATENTS,
            sample.shape[1],
            sample.shape[2],
            device=self.device,
            dtype=sample.dtype,
        )
        intrinsic = (
            torch.eye(3, device=self.device)
            .view(1, 1, 1, 3, 3)
            .repeat(
                1,
                self.max_views,
                OBSERVED_LATENTS,
                1,
                1,
            )
        )
        extrinsic = torch.zeros(
            1, self.max_views, OBSERVED_LATENTS, 3, 4, device=self.device
        )
        extrinsic[..., :3] = torch.eye(3, device=self.device)
        depth = torch.zeros(
            1,
            self.max_views,
            OBSERVED_LATENTS,
            15,
            20,
            device=self.device,
            dtype=torch.float16,
        )
        depth_valid = torch.zeros_like(depth, dtype=torch.bool)
        pixel_valid = torch.zeros_like(depth, dtype=torch.bool)
        for view in range(self.view_count):
            video_latents[0, view] = torch.stack(
                [self.latents[view][index] for index in local_indices],
                dim=1,
            )
            intrinsic[0, view] = torch.from_numpy(
                np.stack([self.intrinsics[view][index] for index in local_indices]),
            ).to(self.device)
            extrinsic[0, view] = torch.from_numpy(
                np.stack([self.extrinsics[view][index] for index in local_indices]),
            ).to(self.device)
            depth[0, view] = torch.from_numpy(
                np.stack([self.depth[view][index] for index in local_indices]),
            ).to(self.device)
            depth_valid[0, view] = torch.from_numpy(
                np.stack([self.depth_valid[view][index] for index in local_indices]),
            ).to(self.device)

            pixel_valid[0, view] = torch.from_numpy(
                np.stack([self.pixel_valid[view][index] for index in local_indices]),
            ).to(self.device)

        view_mask = torch.zeros(1, self.max_views, dtype=torch.bool, device=self.device)
        view_mask[:, : self.view_count] = True
        camera_roles = torch.zeros(1, self.max_views, dtype=torch.long, device=self.device)
        if self.layout == "robocasa365":
            camera_roles[:, 2] = 1
        elif self.layout == "front_wrist":
            camera_roles[:, 1 : self.view_count : 2] = 1
        else:
            camera_roles[:, 1 : self.view_count] = 1
        key = self.prompt_keys[task]
        return {
            "video_latents": video_latents,
            "video_text": self.prompt_tensors[f"{key}.embeddings"][None],
            "video_text_mask": self.prompt_tensors[f"{key}.mask"].bool()[None],
            "video_view_mask": view_mask,
            "video_latent_valid": torch.tensor(
                [valid], device=self.device, dtype=torch.bool
            ),
            "camera_intrinsics": intrinsic,
            "camera_extrinsics": extrinsic,
            "camera_depth": depth,
            "camera_depth_valid": depth_valid,
            "camera_pixel_valid": pixel_valid,
            "camera_roles": camera_roles,
        }
