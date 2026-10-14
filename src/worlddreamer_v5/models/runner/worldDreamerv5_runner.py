# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
from worlddreamer_v5.utils.tensor_reductions import bool_all
from importlib import import_module
import torch
from lightning import LightningModule

from worlddreamer_v5.data.schema import normalize_video_payload
from worlddreamer_v5.models.runner.parameter_groups import NEW_PARAMETER_MARKERS, parameter_groups
from worlddreamer_v5.utils.checkpoint import FORMAT_VERSION, load_model_weights
from worlddreamer_v5.utils.config import build_model
from worlddreamer_v5.utils.schedules import warmup_cosine_scheduler
from worlddreamer_v5.data.action_layout import BASE, GRIPPERS


class WorldDreamerv5Runner(LightningModule):
    def __init__(
        self,
        model_config,
        optimizer,
        scheduler,
        *,
        checkpoint,
        checkpoint_format="worlddreamer-v5",
        expected_contract=None,
        weight_transfer=None,
    ):
        super().__init__()
        self.model_config = dict(model_config)
        self.optimizer_config, self.scheduler_config = dict(optimizer), dict(scheduler)
        self._gradient_check_active = False
        # Materialize before Lightning enters any distributed initialization.
        self.model = build_model(self.model_config)
        metadata = load_model_weights(
            self.model, checkpoint, source_format=checkpoint_format
        )
        if expected_contract is not None:
            from worlddreamer_v5.utils.initialization import initialization_audit
            metadata['robot_action_contract']=normalize_video_payload(metadata.get('robot_action_contract',{}))
            self.initialization_audit=initialization_audit(metadata,expected_contract,weight_transfer)
        else:
            self.initialization_audit=None
        self._register_gradient_checks()

    def configure_optimizers(self):
        config = dict(self.optimizer_config)
        module, name = config.pop("type").rsplit(".", 1)
        parameters = dict(config.get("params", {}))
        groups = parameter_groups(
            self.named_parameters(),
            parameters.get("lr", 1.0),
            parameters.get("weight_decay", 0.1),
        )
        optimizer = getattr(import_module(module), name)(groups, **parameters)
        scheduler = warmup_cosine_scheduler(optimizer, **self.scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        batch=dict(batch)
        contribution=batch.pop('training_weight',None)
        steps = int(self.scheduler_config["num_training_steps"])
        if steps <= 0:
            raise ValueError("scheduler update budget must be positive")
        self.model.set_training_progress(self.global_step / max(steps - 1, 1))
        result = self.model(batch, return_loss=True)
        if contribution is not None:
            if contribution.numel()!=1:raise ValueError('per-device scene batch size must be one')
            result['loss']=result['loss']*contribution.reshape(()).to(result['loss'])
        for name, value in result.items():
            if not bool_all(torch.isfinite(value)):
                raise FloatingPointError(
                    f"non-finite {name} at global_step={self.global_step}"
                )
            self.log(f"train/{name}", value.detach())
        self.log("train/token", batch["input_ids"].shape[1])
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return result

    def _register_gradient_checks(self):
        markers = NEW_PARAMETER_MARKERS + ("video_expert",)
        for name, parameter in self.named_parameters():
            branch = next((value for value in markers if value in name), None)
            if branch is not None and parameter.requires_grad:
                parameter.register_hook(
                    lambda gradient, value=branch: self._record_gradient(
                        value, gradient
                    )
                )

    def on_validation_epoch_start(self):
        self._validation_totals=torch.zeros(7,device=self.device,dtype=torch.float64)

    def validation_step(self,batch,batch_idx):
        # Generate with observed images only; evaluation never gets future latents.
        evaluation=dict(batch)
        evaluation['video_latents']=batch['video_latents'][:,:,:,:3]
        evaluation['video_latent_valid']=batch['video_latent_valid'][:,:3]
        weight=evaluation.pop('validation_weight',torch.ones(1,device=self.device)).sum()
        seed=42+int(batch['record_index'][0])*100000+int(batch['frame_index'][0])
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed);prediction=self.model.generate(evaluation)
        if not bool_all(torch.isfinite(prediction)):raise FloatingPointError('non-finite validation action')
        mask=batch['action_mask'].bool();difference=prediction.float()-batch['action'].float()
        stats=self.trainer.datamodule.train_set.std
        scale=torch.as_tensor(stats,device=self.device,dtype=torch.float32)
        if scale.shape[-1] != self.model.action_shape[-1]:
            raise ValueError(
                f"validation statistics have {scale.shape[-1]} action channels; "
                f"expected {self.model.action_shape[-1]}"
            )
        physical=difference*scale
        active=batch['agent_mask'][:,:,None]
        def mean(value):return (value*active).sum()/active.expand_as(value).sum().clamp_min(1)
        gripper_error = torch.stack(
            [physical[..., index].abs() for index in GRIPPERS], dim=-1
        ).mean(dim=-1)
        values=torch.stack([(difference.square()*mask).sum()/mask.sum().clamp_min(1),
            mean(physical[...,:3].norm(dim=-1)),mean(physical[...,3:6].norm(dim=-1)),
            mean(gripper_error),mean(physical[...,BASE[:2]].norm(dim=-1)),
            mean(physical[...,BASE[2]].abs()),weight.new_tensor(1.)])
        self._validation_totals+=values.double()*weight

    def on_validation_epoch_end(self):
        values=self._validation_totals
        if torch.distributed.is_initialized():torch.distributed.all_reduce(values)
        for i,name in enumerate(('action_mse','ee_position_error_m','ee_rotation_vector_error_rad','gripper_delta_mae','base_translation_velocity_error_m_s','base_yaw_velocity_mae_rad_s')):
            self.log('val/'+name,(values[i]/values[-1].clamp_min(1)).float(),sync_dist=False)

    def _record_gradient(self, branch, gradient):
        if self._gradient_check_active:
            self._seen.add(branch)
            self._finite.append(bool_all(torch.isfinite(gradient)))
        return gradient

    def on_before_backward(self, loss):
        self._gradient_check_active = (
            self.global_step < 10 or self.global_step % 50 == 0
        )
        self._seen, self._finite = set(), []

    def on_after_backward(self):
        if not self._gradient_check_active:
            return
        expected = {key for key in NEW_PARAMETER_MARKERS + ("video_expert",)
                    if any(key in name and value.requires_grad for name,value in self.named_parameters())}
        missing = expected - self._seen
        if missing:
            raise RuntimeError(
                f"missing gradients for model branches: {sorted(missing)}"
            )
        if self._finite and not bool_all(torch.stack(self._finite)):
            raise FloatingPointError(
                f"non-finite gradient at global_step={self.global_step}"
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["worlddreamer_v5_format_version"] = FORMAT_VERSION
        checkpoint["worlddreamer_v5_model_config"] = self.model_config
        checkpoint["worlddreamer_v5_initialization_transfer"] = self.initialization_audit

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("worlddreamer_v5_format_version") != FORMAT_VERSION:
            raise ValueError(
                "optimizer resume requires a worlddreamer-v5 checkpoint; use weight initialization for a legacy model"
            )
        if checkpoint.get("worlddreamer_v5_model_config") != self.model_config:
            raise ValueError("resume architecture configuration mismatch")
