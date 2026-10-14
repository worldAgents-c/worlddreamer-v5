# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
from worlddreamer_v5.utils.tensor_reductions import bool_all

import shutil
import math

import torch
from lightning import Callback


COMMAND_TARGET = "commanded_tcp_pose_and_gripper_drive_v3"
GRADIENT_BRANCHES = (
    "scene_query_base",
    "scene_agent_embedding",
    "agent_query_projector",
    "agent_query_norm",
    "agent_interaction",
    "communication_gates",
)


from worlddreamer_v5.utils.contract import data_contract


class TrainingContractGuard(Callback):
    def __init__(
        self,
        params,
        disk_path,
        min_free_gb=120,
        checkpoint_write_gb=70,
        expected_steps=None,
    ):
        self.params = dict(params)
        self.contract = data_contract(params)
        self.disk_path = disk_path
        self.min_free_gb = min_free_gb
        self.checkpoint_write_gb = checkpoint_write_gb
        self.expected_steps = expected_steps
        self._active = False
        self._handles = []

    def on_fit_start(self, trainer, pl_module):
        if data_contract(self.params) != self.contract:
            raise ValueError("dataset changed after the run was configured")
        self._expected = set()
        for name, parameter in pl_module.named_parameters():
            if not parameter.requires_grad:
                continue
            branch = next(
                (value for value in GRADIENT_BRANCHES if value in name), "pretrained"
            )
            self._expected.add(branch)
            self._handles.append(
                parameter.register_hook(
                    lambda gradient, key=branch: self._record_gradient(key, gradient)
                )
            )
        if self._expected != set(GRADIENT_BRANCHES) | {"pretrained"}:
            raise RuntimeError(f"unexpected worlddreamer-v5 gradient branches: {self._expected}")

    def _record_gradient(self, branch, gradient):
        if self._active:
            self._seen.add(branch)
            self._finite.append(bool_all(torch.isfinite(gradient)))
        return gradient

    def on_before_backward(self, trainer, pl_module, loss):
        self._active = trainer.global_step < 10 or trainer.global_step % 50 == 0
        self._seen, self._finite = set(), []

    def on_train_start(self, trainer, pl_module):
        if self.expected_steps is not None:
            batches = int(trainer.num_training_batches)
            accumulation = int(trainer.accumulate_grad_batches)
            epoch_budget = trainer.max_epochs * math.ceil(batches / accumulation)
            budget = min(trainer.max_steps, epoch_budget) if trainer.max_steps > 0 else epoch_budget
            if budget != self.expected_steps:
                raise ValueError(
                    f"training budget mismatch: {budget} != {self.expected_steps}"
                )
        if trainer.is_global_zero:
            print(
                f"quality_training_start global_step={trainer.global_step} epoch={trainer.current_epoch} expected_steps={self.expected_steps}",
                flush=True,
            )

    def on_train_end(self, trainer, pl_module):
        if (
            self.expected_steps is not None
            and trainer.global_step != self.expected_steps
        ):
            raise RuntimeError(
                f"training step shortfall: {trainer.global_step} != {self.expected_steps}"
            )

    def on_after_backward(self, trainer, pl_module):
        if not self._active:
            return
        if self._seen != self._expected:
            raise RuntimeError(
                f"missing gradients: {sorted(self._expected - self._seen)}"
            )
        if not self._finite or not bool_all(torch.stack(self._finite)):
            raise FloatingPointError(
                f"non-finite gradient at step {trainer.global_step}"
            )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        available = shutil.disk_usage(self.disk_path).free / 2**30
        required = self.min_free_gb + self.checkpoint_write_gb
        if available < required:
            raise RuntimeError(
                f"disk guard before checkpoint: {available:.1f} GiB < {required:.1f} GiB"
            )
        checkpoint["robot_action_contract"] = self.contract

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint.get("robot_action_contract") != self.contract:
            raise ValueError("resume checkpoint dataset/action contract mismatch")

    def teardown(self, trainer, pl_module, stage):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
