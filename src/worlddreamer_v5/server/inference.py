# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import time
import traceback
from types import SimpleNamespace
import numpy as np
from worlddreamer_v5.utils.normalization_registry import load_statistics
from worlddreamer_v5.data.schema import normalize_video_payload
from worlddreamer_v5.server.protocol import (
    ACTION_DIM,
    ACTION_LENGTH,
    MODEL_SEED_PROTOCOL,
    MAX_AGENTS,
    prompt_conversations,
    recv_npz,
    relative_pose_features,
    scalar_int,
    scalar_text,
    send_npz,
)
from worlddreamer_v5.server.streaming import (
    StreamingVideoSession,
    OBSERVED_LATENTS,
    TEMPORAL_COMPRESSION,
)
from worlddreamer_v5.utils.checkpoint import file_sha256, load_model_weights
from worlddreamer_v5.utils.config import build_model, load_config


def validate_inference_contract(config, metadata):
    from worlddreamer_v5.utils.contract import data_contract
    from worlddreamer_v5.utils.data_paths import data_path_mapping, remap_data_paths

    from worlddreamer_v5.utils.model_package import serving_contract

    if "conditioning" in config:
        expected, manifest = serving_contract(config, metadata)
    else:
        expected = data_contract(config["data"])
        actual = normalize_video_payload(metadata.get("robot_action_contract", {}))
        if actual != expected:
            raise ValueError("inference checkpoint/data contract mismatch")
        manifest = json.loads(Path(config["data"]["video_cache_manifest"]).read_text())
        manifest = remap_data_paths(manifest, data_path_mapping(config["data"]))
    if expected["video_observed_latents"] != OBSERVED_LATENTS:
        raise ValueError("streaming observed window differs from training")
    model = config["model"]
    if (
        model.get("video_observed_latents", 3) != OBSERVED_LATENTS
        or model.get("max_agents", 4) != MAX_AGENTS
    ):
        raise ValueError("server tensor layout differs from model configuration")
    if manifest["temporal_compression"] != TEMPORAL_COMPRESSION:
        raise ValueError("streaming temporal compression differs from training")
    return expected, manifest


def build_runtime(config):
    import torch
    from worlddreamer_v5.utils.tensor_reductions import bool_all
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file
    from PIL import Image
    from worlddreamer_v5.data.schema import STATE_ID
    from worlddreamer_v5.data.collator import WorldCollator
    from worlddreamer_v5.data.observation import ObservationTools
    from worlddreamer_v5.utils.normalization import (
        compose_state,
        denormalize_action,
        normalize_quantile,
        resize_image,
        validate_quantiles,
        validate_stats,
    )

    options = config["inference"]
    args = SimpleNamespace(
        device=options.get("device", "cuda:0"),
        normalize=Path(config["data"]["normalize_path"]),
    )
    model = build_model(config["model"])
    metadata = load_model_weights(
        model,
        options["checkpoint"],
        source_format=options.get("checkpoint_format", "worlddreamer-v5"),
    )
    contract, manifest = validate_inference_contract(config, metadata)
    model = model.eval().to(args.device)
    contract = dict(
        contract,
        model_name=model.model_name,
        model_variant=model.variant,
        action_blocks=model.action_layers,
    )
    from worlddreamer_v5.models.variants import validate_asset_config
    validate_asset_config(options["vae_path"], model.architecture["vae"], component="vae")
    vae = (
        AutoencoderKLWan.from_pretrained(
            options["vae_path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to(args.device)
    )
    prompt_tensors = {
        key: value.to(args.device)
        for key, value in load_file(manifest["prompt_cache"], device="cpu").items()
    }
    session = StreamingVideoSession(
        vae, prompt_tensors, manifest["prompt_keys"], args.device
    )
    stats = load_statistics(args.normalize, config["data"].get("normalization_dataset"))
    mean, std = validate_stats(stats["mean"], stats["std"], ACTION_LENGTH)
    q01, q99 = validate_quantiles(stats["q01"], stats["q99"])
    mean = torch.from_numpy(mean).to(args.device)
    std = torch.from_numpy(std).to(args.device)
    collate = WorldCollator(config["data"]["processor_path"])

    def prepare(request: dict[str, np.ndarray], task: str):
        agent_count = scalar_int(request["agent_count"])
        qpos = np.asarray(request["qpos"], dtype=np.float32)
        global_rgb = np.asarray(request["global_rgb"], dtype=np.uint8)
        wrist_rgb = np.asarray(request["wrist_rgb"], dtype=np.uint8)
        world_cameras = np.asarray(request["wrist_cam2world"], dtype=np.float32)
        if qpos.shape != (agent_count, 9):
            raise ValueError(f"qpos shape mismatch: {qpos.shape}")
        if wrist_rgb.shape[0] != agent_count or world_cameras.shape != (
            agent_count,
            4,
            4,
        ):
            raise ValueError(
                f"agent observation mismatch: wrist={wrist_rgb.shape}, cameras={world_cameras.shape}",
            )

        images = [global_rgb, *list(wrist_rgb)]
        images = [
            resize_image(Image.fromarray(image), factor=32, max_pixels=160000)
            for image in images
        ]
        messages = ObservationTools._messages(
            prompt_conversations(task, agent_count), images
        )
        states = np.zeros((MAX_AGENTS, 1, 20), dtype=np.float32)
        agent_mask = np.zeros(MAX_AGENTS, dtype=np.bool_)
        for agent in range(agent_count):
            state = compose_state(
                left_gripper=np.asarray([qpos[agent, 7:9].sum()], dtype=np.float32),
                left_joint=qpos[agent, :7],
                right_gripper=np.zeros(1, dtype=np.float32),
                right_joint=np.zeros(7, dtype=np.float32),
            )
            states[agent] = normalize_quantile(state, q01, q99)
            agent_mask[agent] = True

        action = np.zeros((MAX_AGENTS, ACTION_LENGTH, ACTION_DIM), dtype=np.float32)
        action_mask = np.zeros_like(action, dtype=np.int32)
        action_mask[:agent_count, :, :7] = 1
        item = {
            "messages": messages,
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
            "state": torch.from_numpy(states),
            "agent_mask": torch.from_numpy(agent_mask),
            "relative_pose": torch.from_numpy(relative_pose_features(world_cameras)),
            "task": task,
        }
        batch = collate([item])
        query_count = int((batch["input_ids"] == STATE_ID).sum().item())
        sequence_length = int(batch["input_ids"].shape[-1])
        batch = {
            key: value.to(args.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        batch.update(session.model_inputs(task))
        return batch, action_mask, query_count, sequence_length

    @torch.inference_mode()
    def infer(request: dict[str, np.ndarray]) -> dict[str, object]:
        task = scalar_text(request["task"])
        vae_started = time.perf_counter()
        new_latents = session.ingest(request)
        vae_seconds = time.perf_counter() - vae_started
        batch, action_mask, query_count, sequence_length = prepare(request, task)
        policy_seed = scalar_int(request["policy_seed"])
        mask = torch.from_numpy(action_mask).to(args.device).unsqueeze(0)
        cuda_index = torch.device(args.device).index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        model_started = time.perf_counter()
        with torch.random.fork_rng(devices=[cuda_index], enabled=True):
            torch.manual_seed(policy_seed)
            normalized = model.generate(batch)
        model_seconds = time.perf_counter() - model_started
        if tuple(normalized.shape) != (1, MAX_AGENTS, ACTION_LENGTH, ACTION_DIM):
            raise RuntimeError(f"unexpected model output: {tuple(normalized.shape)}")
        if not bool_all(torch.isfinite(normalized)):
            raise FloatingPointError("model returned a non-finite action")
        physical = denormalize_action(normalized * mask, mean, std) * mask
        agent_count = scalar_int(request["agent_count"])
        return {
            "status": "ok",
            "action": physical[0, :agent_count].float().cpu().numpy(),
            "query_count": query_count,
            "sequence_length": sequence_length,
            "normalized_abs_max": float(normalized.abs().max().item()),
            "latency_seconds": vae_seconds + model_seconds,
            "vae_latency_seconds": vae_seconds,
            "model_latency_seconds": model_seconds,
            "policy_seed": policy_seed,
            "model_seed_base": session.model_seed_base,
            "video_frame_count": session.frame_count,
            "video_latent_count": session.latent_count,
            "video_new_latents": new_latents,
            "future_decoder_enabled": False,
        }

    return infer, session, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    options = config["inference"]
    infer, session, contract = build_runtime(config)
    host, port = (
        args.host or options.get("host", "127.0.0.1"),
        args.port or options.get("port", 10100),
    )
    calibration = Path(options["calibration"])
    runtime = {
        "model": "worlddreamer-v5",
        "model_variant": contract["model_variant"],
        "action_blocks": contract["action_blocks"],
        "checkpoint": str(Path(options["checkpoint"]).resolve()),
        "normalize": str(Path(config["data"]["normalize_path"]).resolve()),
        "calibration": str(calibration.resolve()),
        "calibration_sha256": file_sha256(calibration),
        "action_target": contract["action_target"],
        "video_mode": "streaming_observed_only",
        "model_seed_protocol": MODEL_SEED_PROTOCOL,
        "future_decoder_enabled": False,
    }
    print(f"runtime={json.dumps(runtime, sort_keys=True)}", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"ready host={host} port={port} pid={os.getpid()}", flush=True)
        while True:
            connection, address = server.accept()
            print(f"client_connected={address}", flush=True)
            with connection:
                while True:
                    try:
                        request = recv_npz(connection)
                    except ConnectionError:
                        break
                    except (OSError, ValueError) as error:
                        print(
                            f"client_protocol_error={type(error).__name__}: {error}",
                            flush=True,
                        )
                        break
                    try:
                        command = scalar_text(
                            request.get("command", np.asarray("infer"))
                        )
                        if command == "health":
                            response = {"status": "ok", "message": "ready", **runtime}
                        elif command == "reset":
                            session.reset(
                                scalar_int(request["env_seed"]),
                                scalar_int(request["agent_count"]),
                                scalar_int(request["model_seed_base"]),
                            )
                            response = {
                                "status": "ok",
                                "message": "session_reset",
                                "model_seed_base": session.model_seed_base,
                                "future_decoder_enabled": False,
                            }
                        elif command == "infer":
                            response = infer(request)
                        else:
                            raise ValueError(f"unsupported command: {command}")
                    except Exception as error:
                        traceback.print_exc()
                        response = {
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    send_npz(connection, response)
            session.reset(0, 0, 0)
            print("client_disconnected", flush=True)


if __name__ == "__main__":
    main()
