# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

from common import (
    ACTION_LENGTH,
    MODEL_SEED_PROTOCOL,
    TASKS,
    as_numpy,
    policy_seed_for,
    recv_npz,
    recover_j1_targets,
    scalar_text,
    send_npz,
)
from ik_controller import (
    IKExecutionConfig,
    RobosuiteStyleIKExecutor,
    pose_error,
)


@contextmanager
def isolated_legacy_numpy_rng(seed: int):
    """Scope legacy NumPy seeding to environment construction/reset and restore it."""

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


from action_contract import (
    LEGACY_ACTION_CONTRACT, COMMAND_ACTION_CONTRACT, calibration_for_task,
    command_from_drive_aperture, require_action_contract, file_sha256,
)


class GripperLatch:
    """Infer open/close intent from predicted aperture motion and retain contact force."""

    def __init__(self, close_threshold: float, open_threshold: float):
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.command = 1.0
        self.previous_target: float | None = None

    def begin_chunk(self, current_aperture: float) -> None:
        self.previous_target = float(current_aperture)

    def update(self, target_aperture: float) -> float:
        if self.previous_target is None:
            self.previous_target = float(target_aperture)
        delta = float(target_aperture) - self.previous_target
        if delta <= -self.close_threshold:
            self.command = -1.0
        elif delta >= self.open_threshold:
            self.command = 1.0
        self.previous_target = float(target_aperture)
        return self.command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--env-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--max-replans", type=int, default=30)
    parser.add_argument("--action-target", choices=[LEGACY_ACTION_CONTRACT, COMMAND_ACTION_CONTRACT],
                        default=COMMAND_ACTION_CONTRACT)
    parser.add_argument("--action-horizon", type=int, default=ACTION_LENGTH)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=10100)
    parser.add_argument(
        "--sim-cuda-device",
        default="1",
        help="Physical CUDA device reserved for Sapien sensor rendering; the J1 server uses GPU 0.",
    )
    parser.add_argument("--env-config", type=Path, help="Explicit task/agent camera configuration for this dataset")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )
    parser.add_argument("--ik-attempts", type=int, default=1)
    parser.add_argument("--ik-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--ik-mode",
        choices=("current", "robosuite_style", "robosuite_continuous", "limit_aware"),
        default="current",
        help=(
            "Keep the original executor, enable rate-limited robust IK, or preserve exact "
            "targets while selecting the closest continuous IK solution."
        ),
    )
    parser.add_argument("--ik-max-translation-step-m", type=float, default=0.02)
    parser.add_argument("--ik-max-rotation-step-rad", type=float, default=0.05)
    parser.add_argument("--ik-max-joint-step-rad", type=float, default=0.045)
    parser.add_argument(
        "--ik-backoff-scales",
        type=float,
        nargs="+",
        default=(1.0, 0.5, 0.25, 0.125),
    )
    parser.add_argument("--gripper-close-threshold", type=float, default=2e-3)
    parser.add_argument("--gripper-open-threshold", type=float, default=3e-3)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--show-env-output", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.action_horizon <= ACTION_LENGTH:
        parser.error(f"--action-horizon must be in [1, {ACTION_LENGTH}]")
    if args.max_replans <= 0 or args.ik_attempts <= 0:
        parser.error("--max-replans and --ik-attempts must be positive")
    args.ik_config = None
    if args.ik_mode == "limit_aware" and args.ik_attempts != 1:
        parser.error("limit-aware IK requires --ik-attempts=1; no random restarts")
    if args.ik_mode != "current":
        try:
            args.ik_config = IKExecutionConfig(
                max_translation_step_m=args.ik_max_translation_step_m,
                max_rotation_step_rad=args.ik_max_rotation_step_rad,
                max_joint_step_rad=args.ik_max_joint_step_rad,
                backoff_scales=tuple(args.ik_backoff_scales),
                ik_attempts=args.ik_attempts,
                ik_threshold=args.ik_threshold,
                rate_limit_steps=args.ik_mode == "robosuite_style",
            )
        except ValueError as error:
            parser.error(str(error))
    return args


def success_value(info: dict) -> bool:
    return bool(as_numpy(info.get("success", False)).reshape(-1)[0])


def bool_value(value) -> bool:
    return bool(as_numpy(value).reshape(-1)[0])


def observation_arrays(observation: dict, agent_count: int) -> dict[str, np.ndarray]:
    global_rgb = as_numpy(observation["sensor_data"]["head_camera_global"]["rgb"]).astype(np.uint8)
    wrist_rgb = []
    world_cameras = []
    qpos = []
    for agent in range(agent_count):
        camera = f"wrist_camera_agent{agent}"
        wrist_rgb.append(as_numpy(observation["sensor_data"][camera]["rgb"]).astype(np.uint8))
        world_cameras.append(
            as_numpy(observation["sensor_param"][camera]["cam2world_gl"]).astype(np.float32),
        )
        qpos.append(as_numpy(observation["agent"][f"panda-{agent}"]["qpos"]).astype(np.float32))
    return {
        "global_rgb": global_rgb,
        "wrist_rgb": np.stack(wrist_rgb),
        "wrist_cam2world": np.stack(world_cameras),
        "qpos": np.stack(qpos),
    }


def tiled_rgb(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([arrays["global_rgb"], *list(arrays["wrist_rgb"])], axis=1)


def request_action(
    connection: socket.socket,
    task: str,
    arrays: dict[str, np.ndarray],
    policy_seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    send_npz(
        connection,
        {
            "command": "infer",
            "task": task,
            "agent_count": arrays["qpos"].shape[0],
            "policy_seed": policy_seed,
            **arrays,
        },
    )
    response = recv_npz(connection)
    if scalar_text(response["status"]) != "ok":
        raise RuntimeError(scalar_text(response.get("error", np.asarray("unknown server error"))))
    action = np.asarray(response["action"], dtype=np.float32)
    expected = (arrays["qpos"].shape[0], ACTION_LENGTH, 18)
    if action.shape != expected or not np.isfinite(action).all():
        raise RuntimeError(f"invalid action response: shape={action.shape}, finite={np.isfinite(action).all()}")
    diagnostics = {
        "query_count": int(response["query_count"].reshape(())),
        "sequence_length": int(response["sequence_length"].reshape(())),
        "normalized_abs_max": float(response["normalized_abs_max"].reshape(())),
        "latency_seconds": float(response["latency_seconds"].reshape(())),
    }
    return action, diagnostics


def main() -> None:
    args = parse_args()
    # Simulation dynamics run on CPU, but Sapien still needs one CUDA-visible device
    # for sensor rendering. Keep it physically disjoint from the J1 server's GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.sim_cuda_device
    import robofactory
    repo = Path(robofactory.__file__).resolve().parent
    config = (args.env_config or repo / "configs/worlddreamer" / TASKS[args.task]["config"]).resolve()
    if not config.is_file() or not args.calibration.is_file():
        raise FileNotFoundError(config if not config.is_file() else args.calibration)

    import gymnasium as gym
    import imageio.v2 as imageio
    from transforms3d.quaternions import mat2quat

    import robofactory  # noqa: F401
    from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.resolve() / args.task / f"seed_{args.env_seed:06d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    video_path = output_dir / "global_wrist_hstack.mp4"
    tcp_to_camera = None
    result = {
        "status": "running",
        "task": args.task,
        "env_seed": args.env_seed,
        "policy_seed_base": args.policy_seed,
        "model_seed_protocol": MODEL_SEED_PROTOCOL,
        "max_replans": args.max_replans,
        "action_horizon": args.action_horizon,
        "action_target": args.action_target,
        "env_config": str(config),
        "env_config_sha256": file_sha256(config),
        "calibration_sha256": file_sha256(args.calibration),
        "arm_interpolation": False,
        "ik_controller": {
            "mode": args.ik_mode,
            "attempts": args.ik_attempts,
            "threshold": args.ik_threshold,
            "max_translation_step_m": args.ik_max_translation_step_m,
            "max_rotation_step_rad": args.ik_max_rotation_step_rad,
            "max_joint_step_rad": args.ik_max_joint_step_rad,
            "backoff_scales": list(args.ik_backoff_scales),
            "rate_limit_steps": bool(args.ik_config and args.ik_config.rate_limit_steps),
        },
        "rng_policy": {
            "environment": "explicit reset seed plus restored legacy NumPy scope",
            "policy": "separate process with forked/restored Torch RNG; base seed plus replan",
            "policy_seed_depends_on_env_seed": False,
            "video_latent_seed_depends_on_env_seed": False,
            "global_policy_seed_in_sim_process": False,
            "sim_cuda_visible_devices": args.sim_cuda_device,
            "policy_cuda_visible_devices": "0 (set by the server launcher)",
        },
        "output_dir": str(output_dir),
        "video": str(video_path) if args.record_video else None,
        "replans": [],
    }
    if args.ik_mode == "limit_aware":
        from dataclasses import asdict
        from limit_aware_ik import LimitAwareConfig
        result["ik_controller"]["limit_aware_config"] = asdict(LimitAwareConfig())
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    environment = None
    planner = None
    ik_executors = []
    writer = None
    connection = None
    started = time.time()
    quiet = nullcontext() if args.show_env_output else redirect_stdout(io.StringIO())
    try:
        with isolated_legacy_numpy_rng(args.env_seed), quiet:
            environment = gym.make(
                TASKS[args.task]["env_id"],
                config=str(config),
                obs_mode="rgb+depth",
                control_mode="pd_joint_pos",
                render_mode="sensors",
                sim_backend="cpu",
                sensor_configs={"shader_pack": "default"},
            )
            observation, info = environment.reset(seed=args.env_seed)

        with (nullcontext() if args.show_env_output else redirect_stdout(io.StringIO())):
            planner = PandaArmMotionPlanningSolver(
                environment,
                debug=False,
                vis=False,
                base_pose=[agent.robot.pose for agent in environment.agent.agents],
                visualize_target_grasp_pose=False,
                print_env_info=False,
                is_multi_agent=True,
            )
        agent_count = planner.agent_num
        tcp_to_camera = calibration_for_task(args.calibration, args.task, agent_count)
        if args.ik_config is not None:
            executor_type = RobosuiteStyleIKExecutor
            if args.ik_mode == "limit_aware":
                from limit_aware_ik import LimitAwareIKExecutor
                executor_type = LimitAwareIKExecutor
            ik_executors = [
                executor_type(planner.planner[agent], args.ik_config)
                for agent in range(agent_count)
            ]
        latches = [
            GripperLatch(args.gripper_close_threshold, args.gripper_open_threshold)
            for _ in range(agent_count)
        ]
        connection = socket.create_connection((args.server_host, args.server_port), timeout=60)
        connection.settimeout(1800)
        send_npz(connection, {"command": "health"})
        health = recv_npz(connection)
        if scalar_text(health["status"]) != "ok":
            raise RuntimeError(f"server health failed: {health}")
        require_action_contract(args.action_target, {
            "action_target": scalar_text(health.get("action_target", np.asarray(LEGACY_ACTION_CONTRACT)))
        })
        if "calibration_sha256" in health:
            if scalar_text(health["calibration_sha256"]) != result["calibration_sha256"]:
                raise ValueError("model server and simulator calibration mismatch")
        elif args.action_target == COMMAND_ACTION_CONTRACT:
            raise ValueError("v3 evaluation requires server calibration provenance")
        result["server_checkpoint"] = scalar_text(health["checkpoint"])

        arrays = observation_arrays(observation, agent_count)
        if args.record_video:
            writer = imageio.get_writer(
                video_path,
                fps=args.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            writer.append_data(tiled_rgb(arrays))

        ever_success = success_value(info)
        raw_steps = 0
        ik_successes = 0
        ik_failures = 0
        ik_solver_calls = 0
        ik_backoffs = 0
        ik_translation_limits = 0
        ik_rotation_limits = 0
        ik_joint_limits = 0
        terminated_value = False
        truncated_value = False
        termination_reason = "initial_success" if ever_success else "max_replans"

        for replan in range(args.max_replans):
            if ever_success or terminated_value or truncated_value:
                break
            arrays = observation_arrays(observation, agent_count)
            current_policy_seed = policy_seed_for(args.policy_seed, replan)
            action, diagnostics = request_action(
                connection,
                args.task,
                arrays,
                current_policy_seed,
            )
            if diagnostics["query_count"] != agent_count:
                raise RuntimeError(
                    f"query count {diagnostics['query_count']} != agent count {agent_count}",
                )
            targets = recover_j1_targets(
                action,
                arrays["qpos"],
                arrays["wrist_cam2world"],
                tcp_to_camera,
            )
            for agent, latch in enumerate(latches):
                latch.begin_chunk(float(arrays["qpos"][agent, 7:9].sum()))
                if ik_executors:
                    ik_executors[agent].begin_chunk(arrays["qpos"][agent])

            replan_record = {
                "index": replan,
                "policy_seed": current_policy_seed,
                **diagnostics,
                "executed_actions": 0,
                "ik_successes": 0,
                "ik_failures": 0,
                "ik_solver_calls": 0,
                "ik_backoffs": 0,
                "ik_translation_limits": 0,
                "ik_rotation_limits": 0,
                "ik_joint_limits": 0,
                "max_raw_joint_delta_rad": 0.0,
                "max_commanded_joint_delta_rad": 0.0,
                "max_projection_position_error_m": 0.0,
                "max_projection_rotation_error_rad": 0.0,
                "max_post_step_joint_error_rad": 0.0,
                "max_post_step_position_error_m": 0.0,
                "max_post_step_rotation_error_rad": 0.0,
                "max_translation_delta_m": float(np.linalg.norm(action[:, :, :3], axis=-1).max()),
                "max_rotation_delta_rad": float(np.linalg.norm(action[:, :, 3:6], axis=-1).max()),
                "max_gripper_delta_m": float(np.abs(action[:, :, 6]).max()),
            }

            for action_index in range(args.action_horizon):
                action_dict = {}
                ik_step_results = {}
                current_arrays = observation_arrays(observation, agent_count)
                for agent in range(agent_count):
                    robot_key = f"panda-{agent}"
                    current_qpos = current_arrays["qpos"][agent]
                    world_goal = np.concatenate(
                        [
                            targets["position"][agent, action_index],
                            mat2quat(targets["rotation"][agent, action_index]),
                        ],
                    )
                    base_goal = planner.planner[agent].transform_goal_to_wrt_base(world_goal)
                    if ik_executors:
                        step_result = ik_executors[agent].solve(base_goal, current_qpos)
                        ik_step_results[agent] = step_result
                        arm_target = step_result.arm_target
                        ik_solver_calls += step_result.solver_calls
                        replan_record["ik_solver_calls"] += step_result.solver_calls
                        if step_result.translation_limited:
                            ik_translation_limits += 1
                            replan_record["ik_translation_limits"] += 1
                        if step_result.rotation_limited:
                            ik_rotation_limits += 1
                            replan_record["ik_rotation_limits"] += 1
                        if step_result.joint_limited:
                            ik_joint_limits += 1
                            replan_record["ik_joint_limits"] += 1
                        if step_result.success and step_result.backoff_scale < 1.0:
                            ik_backoffs += 1
                            replan_record["ik_backoffs"] += 1
                        replan_record["max_raw_joint_delta_rad"] = max(
                            replan_record["max_raw_joint_delta_rad"],
                            step_result.raw_max_joint_delta_rad,
                        )
                        replan_record["max_commanded_joint_delta_rad"] = max(
                            replan_record["max_commanded_joint_delta_rad"],
                            step_result.commanded_max_joint_delta_rad,
                        )
                        replan_record["max_projection_position_error_m"] = max(
                            replan_record["max_projection_position_error_m"],
                            step_result.projection_position_error_m,
                        )
                        replan_record["max_projection_rotation_error_rad"] = max(
                            replan_record["max_projection_rotation_error_rad"],
                            step_result.projection_rotation_error_rad,
                        )
                        if step_result.success:
                            ik_successes += 1
                            replan_record["ik_successes"] += 1
                        else:
                            ik_failures += 1
                            replan_record["ik_failures"] += 1
                    else:
                        status, solutions = planner.planner[agent].IK(
                            base_goal,
                            current_qpos,
                            n_init_qpos=args.ik_attempts,
                            threshold=args.ik_threshold,
                        )
                        if status == "Success" and solutions:
                            indices = planner.planner[agent].move_group_joint_indices
                            selected = min(
                                solutions,
                                key=lambda value: np.linalg.norm(
                                    np.asarray(value)[indices] - current_qpos[:7],
                                ),
                            )
                            arm_target = np.asarray(selected, dtype=np.float32)[indices]
                            ik_successes += 1
                            replan_record["ik_successes"] += 1
                        else:
                            arm_target = current_qpos[:7]
                            ik_failures += 1
                            replan_record["ik_failures"] += 1
                    aperture = targets["aperture"][agent, action_index]
                    gripper = (float(command_from_drive_aperture(aperture))
                               if args.action_target == COMMAND_ACTION_CONTRACT
                               else latches[agent].update(aperture))
                    action_dict[robot_key] = np.concatenate(
                        [arm_target, np.asarray([gripper], dtype=np.float32)],
                    )

                step_context = nullcontext() if args.show_env_output else redirect_stdout(io.StringIO())
                with step_context:
                    observation, _, terminated, truncated, info = environment.step(action_dict)
                raw_steps += 1
                replan_record["executed_actions"] += 1
                post_arrays = None
                if ik_step_results:
                    post_arrays = observation_arrays(observation, agent_count)
                    for agent, step_result in ik_step_results.items():
                        post_qpos = post_arrays["qpos"][agent]
                        joint_error = float(
                            np.max(np.abs(post_qpos[:7] - step_result.arm_target)),
                        )
                        actual_pose = ik_executors[agent].pose_for_qpos(post_qpos)
                        position_error, rotation_error = pose_error(
                            actual_pose,
                            step_result.commanded_pose,
                        )
                        replan_record["max_post_step_joint_error_rad"] = max(
                            replan_record["max_post_step_joint_error_rad"],
                            joint_error,
                        )
                        replan_record["max_post_step_position_error_m"] = max(
                            replan_record["max_post_step_position_error_m"],
                            position_error,
                        )
                        replan_record["max_post_step_rotation_error_rad"] = max(
                            replan_record["max_post_step_rotation_error_rad"],
                            rotation_error,
                        )
                ever_success = ever_success or success_value(info)
                terminated_value = bool_value(terminated)
                truncated_value = bool_value(truncated)
                if writer is not None:
                    if post_arrays is None:
                        post_arrays = observation_arrays(observation, agent_count)
                    writer.append_data(tiled_rgb(post_arrays))
                if ever_success:
                    termination_reason = "success"
                    break
                if terminated_value:
                    termination_reason = "terminated"
                    break
                if truncated_value:
                    termination_reason = "truncated"
                    break

            result["replans"].append(replan_record)
            result_path.write_text(json.dumps(result, indent=2) + "\n")

        result.update(
            {
                "status": "complete",
                "success": ever_success,
                "termination_reason": termination_reason,
                "raw_steps": raw_steps,
                "replan_count": len(result["replans"]),
                "ik_successes": ik_successes,
                "ik_failures": ik_failures,
                "ik_success_rate": ik_successes / max(1, ik_successes + ik_failures),
                "ik_solver_calls": ik_solver_calls,
                "ik_backoffs": ik_backoffs,
                "ik_translation_limits": ik_translation_limits,
                "ik_rotation_limits": ik_rotation_limits,
                "ik_joint_limits": ik_joint_limits,
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )
        raise
    finally:
        if writer is not None:
            writer.close()
        if connection is not None:
            connection.close()
        if planner is not None:
            try:
                planner.close()
            except Exception:
                pass
        if environment is not None:
            environment.close()
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
