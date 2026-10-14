# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from worlddreamer_v5.models.trifield import prefix_asymmetric_mask
from .latent_decoder import decode_latent_stream
from .execution import encode_streams, latent_grid


class VideoExpert(nn.Module):
    """Encode observed views and optionally predict their future latent frames."""

    def __init__(self, config_path=None, backbone=None, observed_latents=3, *, config=None):
        super().__init__()
        if backbone is None:
            if config_path is None and config is None:
                raise ValueError(
                    "config or config_path is required when backbone is not provided"
                )
            from worlddreamer_v5.models.video_expert.backbone import VideoTransformer
            from worlddreamer_v5.utils.checkpoint import read_video_config

            model_config = config if config is not None else read_video_config(config_path)
            backbone = VideoTransformer.from_config(model_config).to(torch.bfloat16)
        self.backbone = backbone
        self.observed_latents = int(observed_latents)
        self.backbone.enable_gradient_checkpointing()

    def encode_and_predict(self, hidden_states, timestep, text, text_mask, run_decoder):
        backbone = self.backbone
        patch_grid = latent_grid(backbone, hidden_states)
        tokens_per_frame = patch_grid[1] * patch_grid[2]
        observed_tokens = min(self.observed_latents, patch_grid[0]) * tokens_per_frame
        future_tokens = patch_grid[0] * tokens_per_frame - observed_tokens

        rotary = backbone.rope(hidden_states)
        time_embedding = backbone.time_text_embed(timestep)
        video = backbone.x_embedder(hidden_states)
        context = backbone.context_embedder(text)
        generation_stream = (
            video.clone()
            if run_decoder and backbone.num_generation_layers
            else None
        )
        attention_mask = prefix_asymmetric_mask(observed_tokens, future_tokens, text_mask)

        video, context = encode_streams(
            backbone, video, context, time_embedding, attention_mask, rotary
        )

        encoded_scene = video
        prediction = None
        if generation_stream is not None:
            prediction = decode_latent_stream(
                backbone,
                generation_stream,
                encoded_scene,
                time_embedding,
                rotary,
                patch_grid,
            )

        return encoded_scene, prediction, patch_grid

    def forward(
        self,
        latents,
        text,
        text_mask,
        view_mask,
        latent_valid=None,
        flow_timestep: Optional[torch.Tensor] = None,
        run_decoder=True,
    ):
        if latents.ndim != 6:
            raise ValueError(
                f"expected [B,V,C,T,H,W] latents, got {tuple(latents.shape)}"
            )
        batch_size, view_count, channels, latent_frames, height, width = latents.shape
        if channels != 16 or self.observed_latents > latent_frames:
            raise ValueError(
                "Video expert expects 16-channel latents with a non-empty observed prefix"
            )
        if view_mask.shape != (batch_size, view_count):
            raise ValueError("view_mask shape does not match latent views")

        active_views = view_mask.bool()
        view_indices = active_views.nonzero(as_tuple=False)
        if not len(view_indices):
            raise ValueError("each batch must contain at least one valid camera view")
        selected = latents[active_views]
        selected_text = text[view_indices[:, 0]]
        selected_text_mask = text_mask[view_indices[:, 0]]

        if flow_timestep is None:
            flow_timestep = torch.rand(
                batch_size, device=latents.device, dtype=torch.float32
            )
        if flow_timestep.shape != (batch_size,):
            raise ValueError(
                f"expected one flow timestep per scene, got {tuple(flow_timestep.shape)}"
            )
        selected_timestep = flow_timestep[view_indices[:, 0]]
        hidden_states, noise = self._condition_future(selected, selected_timestep)

        encoded, prediction, grid = self.encode_and_predict(
            hidden_states,
            selected_timestep * 1000.0,
            selected_text,
            selected_text_mask,
            run_decoder=bool(run_decoder and latent_frames > self.observed_latents),
        )
        observed = encoded.view(len(selected), grid[0], grid[1] * grid[2], -1)[
            :, : self.observed_latents
        ]
        padded_observed = self._restore_views(observed, active_views)
        future_loss = observed.sum() * 0.0
        if prediction is not None:
            validity = None if latent_valid is None else latent_valid[view_indices[:, 0]]
            future_loss = self._future_error(prediction, noise - selected, validity)

        return {
            "scene_tokens": padded_observed,
            "future_loss": future_loss,
            "flow_timestep": flow_timestep,
            "patch_grid": grid[1:],
        }

    def _condition_future(self, selected, selected_timestep):
        noise = torch.randn_like(selected)
        noisy = selected.clone()
        fraction = selected_timestep[:, None, None, None, None].to(selected.dtype)
        observed_slice = slice(None, self.observed_latents)
        future_slice = slice(self.observed_latents, None)
        noisy[:, :, future_slice] = (1.0 - fraction) * selected[
            :, :, future_slice
        ] + fraction * noise[:, :, future_slice]
        condition = torch.zeros_like(selected)
        condition[:, :, observed_slice] = selected[:, :, observed_slice]
        condition_mask = selected.new_zeros(
            len(selected), 1, *selected.shape[-3:]
        )
        condition_mask[:, :, observed_slice] = 1.0
        return torch.cat([noisy, condition, condition_mask], dim=1), noise


    @staticmethod
    def _restore_views(observed, active_views):
        padded = observed.new_zeros(*active_views.shape, *observed.shape[1:])
        padded[active_views] = observed
        return padded

    def _future_error(self, prediction, target, validity):
        future = slice(self.observed_latents, None)
        error = F.mse_loss(
            prediction[:, :, future].float(),
            target[:, :, future].float(),
            reduction="none",
        )
        if validity is None:
            return error.mean()
        future_valid = validity[:, future]
        error = error * future_valid[:, None, :, None, None].to(error.dtype)
        channels, height, width = target.shape[1], target.shape[-2], target.shape[-1]
        denominator = future_valid.sum().clamp_min(1) * channels * height * width
        return error.sum() / denominator
