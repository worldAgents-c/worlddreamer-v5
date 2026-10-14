# Copyright (C) 2026 World Agents Corporation.
from .execution import project_latents, run_stream_blocks


def decode_latent_stream(
    model, generation_stream, encoded_scene, time_embedding, rotary, grid
):
    scene_boundary = len(model.fusion_blocks) - model.num_generation_layers
    generated, _ = run_stream_blocks(
        model,
        model.fusion_blocks[scene_boundary:],
        generation_stream,
        encoded_scene,
        time_embedding,
        None,
        rotary,
    )
    return project_latents(model, generated, time_embedding, grid)
