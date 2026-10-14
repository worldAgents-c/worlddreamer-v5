# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

from io import BytesIO
import socket
import struct
from typing import Mapping

import numpy as np


ACTION_LENGTH = 30
from worlddreamer_v5.data.action_layout import ACTION_DIM
MAX_AGENTS = 4
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MODEL_SEED_MODULUS = 2**31 - 1
MODEL_SEED_PROTOCOL = "fixed_replan_v1"
LATENT_SEED_VIEW_STRIDE = 100

TASKS = {
    "PlaceFood": {
        "env_id": "PlaceFood-rf",
        "config": "place_food.yaml",
        "instruction": "place the food into the pot.",
    },
    "CameraAlignment": {
        "env_id": "CameraAlignment-rf",
        "config": "camera_alignment.yaml",
        "instruction": "lift and align the camera with the food.",
    },
    "TakePhoto": {
        "env_id": "TakePhoto-rf",
        "config": "take_photo.yaml",
        "instruction": "hold the camera and food in position, then press the camera button to take a photo.",
    },
}


def policy_seed_for(base_seed: int, replan: int) -> int:
    """Return a reproducible action-noise schedule independent of the environment seed."""

    if replan < 0:
        raise ValueError("replan must be non-negative")
    return int((int(base_seed) + int(replan)) % MODEL_SEED_MODULUS)


def latent_seed_for(base_seed: int, view_index: int) -> int:
    """Return a per-view VAE seed independent of the environment seed."""

    if view_index < 0:
        raise ValueError("view_index must be non-negative")
    return int(
        (int(base_seed) * LATENT_SEED_VIEW_STRIDE + int(view_index))
        % MODEL_SEED_MODULUS
    )


def _recv_all(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        packet = connection.recv(length - len(chunks))
        if not packet:
            raise ConnectionError("connection closed while receiving a message")
        chunks.extend(packet)
    return bytes(chunks)


def send_npz(connection: socket.socket, payload: Mapping[str, object]) -> None:
    arrays = {}
    for key, value in payload.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            value = (
                value.float().numpy()
                if str(value.dtype) == "torch.bfloat16"
                else value.numpy()
            )
        arrays[key] = np.asarray(value)
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message is too large: {len(data)} bytes")
    connection.sendall(struct.pack(">I", len(data)) + data)


def recv_npz(connection: socket.socket) -> dict[str, np.ndarray]:
    size = struct.unpack(">I", _recv_all(connection, 4))[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"refusing oversized message: {size} bytes")
    with np.load(BytesIO(_recv_all(connection, size)), allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def scalar_text(value: np.ndarray | object) -> str:
    return str(np.asarray(value).reshape(()).item())


def scalar_int(value: np.ndarray | object) -> int:
    return int(np.asarray(value).reshape(()).item())


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim and value.shape[0] == 1:
        value = value[0]
    return np.asarray(value)


def prompt_conversations(task: str, agent_count: int) -> list[dict[str, str]]:
    if task not in TASKS:
        raise KeyError(f"unsupported task: {task}")
    if not 1 <= agent_count <= MAX_AGENTS:
        raise ValueError(f"agent_count={agent_count} is outside [1, {MAX_AGENTS}]")
    blocks = [
        f"Task instruction: All robot agents should collaboratively {TASKS[task]['instruction']}",
        "The following synchronized observations are captured from multiple single-arm robot agents operating in the same scene.",
        "Global View:\n<image>",
    ]
    blocks.extend(
        f"Agent {index + 1} Wrist View:\n<image>" for index in range(agent_count)
    )
    human = "\n\n".join(blocks) + " /no_cot"
    assistant = "<cot></cot>" + "<state>" * agent_count
    return [
        {"from": "human", "value": human},
        {"from": "gpt", "value": assistant},
    ]


def relative_pose_features(
    world_cameras: np.ndarray, max_agents: int = MAX_AGENTS
) -> np.ndarray:
    transforms = np.asarray(world_cameras, dtype=np.float32)
    agent_count = transforms.shape[0]
    if transforms.shape != (agent_count, 4, 4) or agent_count > max_agents:
        raise ValueError(f"invalid camera transforms: {transforms.shape}")
    features = np.zeros((max_agents, max_agents, 10), dtype=np.float32)
    for source in range(agent_count):
        source_rotation = transforms[source, :3, :3]
        source_position = transforms[source, :3, 3]
        for target in range(agent_count):
            target_rotation = transforms[target, :3, :3]
            target_position = transforms[target, :3, 3]
            translation = source_rotation.T @ (target_position - source_position)
            rotation = source_rotation.T @ target_rotation
            rotation_6d = rotation[:, :2].T.reshape(-1)
            features[source, target] = np.concatenate(
                [translation, rotation_6d, [np.linalg.norm(translation)]],
            )
    return features
