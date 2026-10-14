# Copyright (C) 2026 World Agents Corporation.
import torch
from torch import nn
from diffusers.models.embeddings import (
    Timesteps,
    TimestepEmbedding,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_utils import get_parameter_dtype


class VideoPatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        kernel = (
            (patch_size,) * 3
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel, stride=kernel
        )

    def forward(self, latents):
        encoded = self.proj(latents)
        batch, channels = encoded.shape[:2]
        return encoded.reshape(batch, channels, -1).transpose(1, 2)


class FlowConditionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep):
        phase = self.time_proj(timestep)
        parameter_dtype = get_parameter_dtype(self.timestep_embedder)
        if parameter_dtype in (torch.float16, torch.bfloat16):
            phase = phase.to(parameter_dtype)
        return self.timestep_embedder(phase)


class VideoRotaryEmbedding(nn.Module):
    def __init__(self, patch_size, patch_size_t, rope_dim, theta=256.0):
        super().__init__()
        self.patch_size, self.patch_size_t = patch_size, patch_size_t
        self.rope_dim, self.theta = rope_dim, theta

    def forward(self, latents):
        strides = (self.patch_size_t, self.patch_size, self.patch_size)
        axes = []
        for extent, stride in zip(latents.shape[-3:], strides):
            axes.append(
                torch.arange(
                    extent // stride,
                    device=latents.device,
                    dtype=torch.float32,
                )
            )
        mesh = torch.meshgrid(*axes, indexing="ij")
        freq_dtype = (
            torch.float32 if latents.device.type in {"mps", "npu"} else torch.float64
        )
        components = []
        for width, axis in zip(self.rope_dim, mesh):
            components.append(
                get_1d_rotary_pos_embed(
                    dim=width,
                    pos=axis.flatten(),
                    theta=self.theta,
                    use_real=True,
                    freqs_dtype=freq_dtype,
                )
            )
        return tuple(torch.cat(parts, dim=1) for parts in zip(*components))


class ImageConditionProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, in_features)
        self.act_fn = nn.GELU()
        self.linear_2 = nn.Linear(in_features, hidden_size)
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(self, images):
        features = self.norm_in(images)
        features = self.linear_1(features)
        features = self.act_fn(features)
        features = self.linear_2(features)
        return self.norm_out(features)
