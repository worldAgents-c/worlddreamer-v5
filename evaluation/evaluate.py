# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import numpy as np

import rollout as baseline
from common import ACTION_LENGTH, recv_npz, scalar_text, send_npz


class ObservationHistory:
    def __init__(self):
        self.reset(0)

    def reset(self, env_seed: int) -> None:
        self.env_seed = int(env_seed)
        self.agent_count = 0
        self.sent_frames = 0
        self.server_reset = False
        self.model_seed_base: int | None = None
        self.rgb: list[np.ndarray] = []
        self.intrinsics: list[np.ndarray] = []
        self.extrinsics: list[np.ndarray] = []
        self.depth: list[np.ndarray] = []
        self.depth_valid: list[np.ndarray] = []

    @staticmethod
    def _view_names(agent_count: int) -> list[str]:
        return ["head_camera_global"] + [f"wrist_camera_agent{index}" for index in range(agent_count)]

    def append(self, observation: dict) -> None:
        agent_count = len([key for key in observation["agent"] if key.startswith("panda-")])
        if not agent_count:
            raise RuntimeError("RoboFactory observation contains no panda agents")
        if self.agent_count and agent_count != self.agent_count:
            raise RuntimeError(f"agent count changed from {self.agent_count} to {agent_count}")
        self.agent_count = agent_count

        rgb, intrinsics, extrinsics, depth, depth_valid = [], [], [], [], []
        for camera in self._view_names(agent_count):
            sensor = observation["sensor_data"][camera]
            parameters = observation["sensor_param"][camera]
            rgb.append(baseline.as_numpy(sensor["rgb"]).astype(np.uint8))
            intrinsics.append(baseline.as_numpy(parameters["intrinsic_cv"]).astype(np.float32))
            extrinsics.append(baseline.as_numpy(parameters["extrinsic_cv"]).astype(np.float32))
            metric_depth = baseline.as_numpy(sensor["depth"]).astype(np.float32)
            if metric_depth.ndim == 3 and metric_depth.shape[-1] == 1:
                metric_depth = metric_depth[..., 0]
            if metric_depth.ndim != 2:
                raise ValueError(f"unexpected {camera} depth shape: {metric_depth.shape}")
            metric_depth = metric_depth[8::16, 8::16]
            valid = metric_depth > 0
            depth.append(np.where(valid, metric_depth / 1000.0, 0.0).astype(np.float16))
            depth_valid.append(valid)

        frame_shapes = {tuple(value.shape) for value in rgb}
        if len(frame_shapes) != 1:
            raise ValueError(f"camera RGB shapes differ: {sorted(frame_shapes)}")
        self.rgb.append(np.stack(rgb))
        self.intrinsics.append(np.stack(intrinsics))
        self.extrinsics.append(np.stack(extrinsics))
        self.depth.append(np.stack(depth))
        self.depth_valid.append(np.stack(depth_valid))

    def payload(self) -> dict[str, np.ndarray | int]:
        if not self.rgb:
            raise RuntimeError("no unsent observation frames")
        return {
            "video_frame_start": self.sent_frames,
            "video_rgb": np.stack(self.rgb, axis=1),
            "video_intrinsics": np.stack(self.intrinsics, axis=1),
            "video_extrinsics": np.stack(self.extrinsics, axis=1),
            "video_depth": np.stack(self.depth, axis=1),
            "video_depth_valid": np.stack(self.depth_valid, axis=1),
        }

    def commit(self) -> None:
        count = len(self.rgb)
        self.sent_frames += count
        self.rgb.clear()
        self.intrinsics.clear()
        self.extrinsics.clear()
        self.depth.clear()
        self.depth_valid.clear()


HISTORY = ObservationHistory()


def request_action(
    connection,
    task: str,
    arrays: dict[str, np.ndarray],
    policy_seed: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    agent_count = int(arrays["qpos"].shape[0])
    if not HISTORY.server_reset:
        HISTORY.model_seed_base = int(policy_seed)
        send_npz(
            connection,
            {
                "command": "reset",
                "env_seed": HISTORY.env_seed,
                "agent_count": agent_count,
                "model_seed_base": HISTORY.model_seed_base,
            },
        )
        response = recv_npz(connection)
        if scalar_text(response["status"]) != "ok":
            raise RuntimeError(scalar_text(response.get("error", np.asarray("server reset failed"))))
        if bool(np.asarray(response["future_decoder_enabled"]).reshape(())):
            raise RuntimeError("future decoder must be disabled during evaluation")
        if int(np.asarray(response["model_seed_base"]).reshape(())) != HISTORY.model_seed_base:
            raise RuntimeError("server reset used an unexpected model seed")
        HISTORY.server_reset = True

    send_npz(
        connection,
        {
            "command": "infer",
            "task": task,
            "agent_count": agent_count,
            "policy_seed": policy_seed,
            **arrays,
            **HISTORY.payload(),
        },
    )
    response = recv_npz(connection)
    if scalar_text(response["status"]) != "ok":
        raise RuntimeError(scalar_text(response.get("error", np.asarray("unknown server error"))))
    if bool(np.asarray(response["future_decoder_enabled"]).reshape(())):
        raise RuntimeError("server unexpectedly enabled the future decoder")

    action = np.asarray(response["action"], dtype=np.float32)
    expected = (agent_count, ACTION_LENGTH, 18)
    if action.shape != expected or not np.isfinite(action).all():
        raise RuntimeError(f"invalid action response: shape={action.shape}, finite={np.isfinite(action).all()}")
    HISTORY.commit()
    diagnostics = {
        "query_count": int(response["query_count"].reshape(())),
        "sequence_length": int(response["sequence_length"].reshape(())),
        "normalized_abs_max": float(response["normalized_abs_max"].reshape(())),
        "latency_seconds": float(response["latency_seconds"].reshape(())),
        "vae_latency_seconds": float(response["vae_latency_seconds"].reshape(())),
        "model_latency_seconds": float(response["model_latency_seconds"].reshape(())),
        "video_frame_count": int(response["video_frame_count"].reshape(())),
        "video_latent_count": int(response["video_latent_count"].reshape(())),
        "video_new_latents": int(response["video_new_latents"].reshape(())),
        "video_latent_seed_base": int(response["model_seed_base"].reshape(())),
        "future_decoder_enabled": False,
    }
    return action, diagnostics


def install_observation_hooks() -> None:
    import gymnasium as gym

    original_make = gym.make

    def make_with_history(*args, **kwargs):
        environment = original_make(*args, **kwargs)
        original_reset = environment.reset
        original_step = environment.step

        def reset(*reset_args, **reset_kwargs):
            observation, info = original_reset(*reset_args, **reset_kwargs)
            HISTORY.reset(int(reset_kwargs.get("seed", 0)))
            HISTORY.append(observation)
            return observation, info

        def step(action):
            observation, reward, terminated, truncated, info = original_step(action)
            HISTORY.append(observation)
            return observation, reward, terminated, truncated, info

        environment.reset = reset
        environment.step = step
        return environment

    gym.make = make_with_history
    baseline.request_action = request_action


if __name__ == "__main__":
    install_observation_hooks()
    baseline.main()
