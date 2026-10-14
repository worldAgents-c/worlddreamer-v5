# Copyright (C) 2026 World Agents Corporation.
from itertools import chain

import torch


def latent_grid(model, latents):
    temporal_patch = int(model.config.patch_size_t)
    spatial_patch = int(model.config.patch_size)
    return tuple(
        size // patch
        for size, patch in zip(
            latents.shape[-3:],
            (temporal_patch, spatial_patch, spatial_patch),
        )
    )


def run_stream_blocks(
    model, blocks, video, context, conditioning, mask, rotary, image_tokens=0
):
    for layer in blocks:
        layer_inputs = (video, context, conditioning, mask, rotary, image_tokens)
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            video, context = model._gradient_checkpointing_func(layer, *layer_inputs)
        else:
            video, context = layer(*layer_inputs)
    return video, context


def encode_streams(model, video, context, conditioning, mask, rotary, image_tokens=0):
    scene_boundary = len(model.fusion_blocks) - model.num_generation_layers
    stages = chain(
        model.exchange_blocks,
        model.fusion_blocks[:scene_boundary],
    )
    return run_stream_blocks(
        model, stages, video, context, conditioning, mask, rotary, image_tokens
    )


def project_latents(model, tokens, conditioning, grid):
    decoded = model.proj_out(model.norm_out(tokens, conditioning))
    spatial_patch = int(model.config.patch_size)
    temporal_patch = int(model.config.patch_size_t)
    patch_volume = decoded.reshape(
        tokens.shape[0],
        *grid,
        -1,
        temporal_patch,
        spatial_patch,
        spatial_patch,
    )
    ordered = patch_volume.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return ordered.reshape(
        tokens.shape[0],
        -1,
        grid[0] * temporal_patch,
        grid[1] * spatial_patch,
        grid[2] * spatial_patch,
    )
