# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
from worlddreamer_v5.utils.tensor_reductions import bool_any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributions import Beta
from transformers import Qwen3VLConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from worlddreamer_v5.data.schema import STATE_ID
from worlddreamer_v5.models.vlm.backbone import Qwen3VLForConditionalGeneration
from worlddreamer_v5.models.vlm.scene_queries import AgentInteraction
from worlddreamer_v5.models.action_dreamer import ActionDreamer, Projector, TimestepEmbedder
from worlddreamer_v5.models.video_expert.temporal_encoder import VideoExpert
from worlddreamer_v5.models.trifield import (
    AgentFieldInjector,
    CameraRaySampler,
    FieldRenderHead,
    TriPlaneBuilder,
)
from worlddreamer_v5.utils.precision import bf16_batch
from worlddreamer_v5.data.action_layout import BASE
from worlddreamer_v5.models.control_profiles import control_profile
from worlddreamer_v5.models.variants import MODEL_NAME, MODEL_VARIANT, get_model_profile, validate_asset_config


class _MultiAgentActionContext:
    """Apply scene-specific conditioning around each action block."""

    def __init__(
        self,
        model,
        batch_repeat,
        agent_count,
        workspace,
        agent_positions,
        relative_pose,
        agent_mask,
        action_timestep,
    ):
        self.model = model
        self.batch_repeat = batch_repeat
        self.agent_count = agent_count
        self.workspace = workspace
        self.agent_positions = agent_positions
        self.relative_pose = relative_pose
        self.agent_mask = agent_mask
        self.action_timestep = action_timestep

    def _unflatten(self, tokens):
        if tokens.shape[0] != self.batch_repeat * self.agent_count:
            raise ValueError(
                "action layer batch does not match the scene and agent dimensions"
            )
        return tokens.reshape(self.batch_repeat, self.agent_count, tokens.shape[1], tokens.shape[2])

    def before_block(self, tokens, layer_index):
        hidden_states = self._unflatten(tokens)
        query_index = self.model.query_index
        queries = self.model.agent_field_injector(
            hidden_states[:, :, query_index],
            self.workspace,
            self.agent_positions,
            self.agent_mask,
            self.action_timestep,
            layer_index,
        )
        if self.model.control["query_suffix"]:
            action_start = self.model.action_start
            gate = torch.tanh(self.model.field_action_gates[layer_index]).to(queries.dtype)
            hidden_states = torch.cat((
                hidden_states[:, :, :action_start],
                hidden_states[:, :, action_start:query_index] + (gate * queries).unsqueeze(2),
                queries.unsqueeze(2),
            ), dim=2)
        else:
            hidden_states = torch.cat((hidden_states[:, :, :query_index],
                                      queries.unsqueeze(2), hidden_states[:, :, query_index + 1:]), dim=2)
        return hidden_states.reshape(-1, hidden_states.shape[2], hidden_states.shape[3])

    def after_block(self, tokens, layer_index):
        hidden_states = self._unflatten(tokens)
        query_index = self.model.query_index
        queries = hidden_states[:, :, query_index]
        communicated = self.model.agent_interaction(
            queries, self.relative_pose, self.agent_mask
        )
        gate = torch.tanh(self.model.communication_gates[layer_index]).to(
            communicated.dtype
        )
        hidden_states = torch.cat(
            [
                hidden_states[:, :, :query_index],
                (queries + gate * communicated).unsqueeze(2),
                hidden_states[:, :, query_index + 1 :],
            ],
            dim=2,
        )
        return hidden_states.reshape(-1, hidden_states.shape[2], hidden_states.shape[3])


class WorldDreamerv5(nn.Module):
    """worlddreamer-v5 model; all persistent tensors belong to this entry point."""

    MODEL_NAME = MODEL_NAME

    VIDEO_KEYS = (
        "video_latents",
        "video_text",
        "video_text_mask",
        "video_view_mask",
        "camera_intrinsics",
        "camera_extrinsics",
        "camera_roles",
    )

    def __init__(
        self,
        *,
        vlm_config_path=None,
        video_config_path=None,
        video_observed_latents=None,
        field_size=None,
        field_resolution=None,
        field_samples_per_ray=None,
        field_bounds=None,
        depth_dropout=0.25,
        future_coefficient=0.2,
        field_coefficient_start=0.1,
        field_coefficient_end=0.02,
        max_agents=None,
        communication_gate_init=0.05,
        freq_coefficient=1.0,
        freq_excluded_dims=None,
        ffn_gradient_checkpointing=True,
        async_train=True,
        attention_backend="flash_attention_2",
        state_dim=20,
        action_dim=18,
        state_history_length=1,
        action_length=30,
        model_name=MODEL_NAME,
        variant=MODEL_VARIANT,
    ):
        super().__init__()
        if model_name != MODEL_NAME:
            raise ValueError(f"expected model_name={MODEL_NAME!r}, got {model_name!r}")
        self.architecture = get_model_profile(variant)
        self.model_name = MODEL_NAME
        self.variant = variant
        self.control = control_profile(state_dim, action_dim, state_history_length, action_length)
        self.architecture["action"].update({k: v for k, v in self.control.items()
                                            if k in self.architecture["action"]})
        action = self.architecture["action"]
        self.action_layers = action["layers"]
        self.action_hidden_size = action["hidden_size"]
        if self.control["query_suffix"] and async_train:
            raise ValueError("RoboCasa control requires async_train=False")
        geometry = self.architecture["geometry"]
        supplied = dict(video_observed_latents=video_observed_latents,
                        field_size=field_size, field_resolution=field_resolution,
                        field_samples_per_ray=field_samples_per_ray, max_agents=max_agents)
        for key, value in supplied.items():
            if value is not None and value != geometry[key]:
                raise ValueError(f"{variant} owns {key}={geometry[key]!r}; got {value!r}")
            setattr(self, key, geometry[key])
        if vlm_config_path is not None:
            validate_asset_config(vlm_config_path, self.architecture["vlm"], component="vlm")
        if video_config_path is not None:
            validate_asset_config(video_config_path, self.architecture["video"], component="video")
        self.field_bounds = field_bounds
        self.depth_dropout = float(depth_dropout)
        self.future_coefficient = float(future_coefficient)
        self.field_coefficient_start = float(field_coefficient_start)
        self.field_coefficient_end = float(field_coefficient_end)
        self.communication_gate_init = float(communication_gate_init)
        self.state_shape = (action["state_steps"], action["state_dimensions"])
        self.action_shape = (action["horizon"], action["dimensions"])
        self.action_start = 1 + self.state_shape[0] + int(not self.control["query_suffix"])
        self.query_index = (1 + self.state_shape[0] + self.action_shape[0]
                            if self.control["query_suffix"] else 1 + self.state_shape[0])
        self.freq_coefficient = float(freq_coefficient)
        self.freq_excluded_dims = list(
            BASE if freq_excluded_dims is None else freq_excluded_dims
        )
        self.ffn_gradient_checkpointing = bool(ffn_gradient_checkpointing)
        self.async_train = bool(async_train)
        self.training_repeat, self.num_steps = action["training_repeat"], action["flow_steps"]
        self.beta = Beta(1.5, 1.0)
        self._build_language_backbone(attention_backend)
        self._build_action_modules()
        self._build_world_modules()
        self.to(torch.bfloat16)

    def _build_language_backbone(self, attention_backend):
        config = Qwen3VLConfig(**self.architecture["vlm"])
        self.vision_language_backbone = Qwen3VLForConditionalGeneration._from_config(
            config,
            attn_implementation=attention_backend,
            dtype=torch.bfloat16,
        ).train()
        self.vision_language_backbone.model.get_input_embeddings().requires_grad_(False)
        # DDP unused-parameter discovery must see the visual autograd graph.
        # Reentrant checkpointing hides it on the first pass and can mark a
        # parameter ready twice during backward, especially with async branches.
        self.vision_language_backbone.model.visual.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if self.ffn_gradient_checkpointing:
            for layer in self.vision_language_backbone.model.language_model.layers:
                original_forward = layer.mlp.forward

                def checkpointed_forward(x, forward=original_forward, module=layer.mlp):
                    if module.training and torch.is_grad_enabled():
                        return torch.utils.checkpoint.checkpoint(
                            forward, x, use_reentrant=False
                        )
                    return forward(x)

                layer.mlp.forward = checkpointed_forward
        for module in (
            self.vision_language_backbone.model.action_embed,
            self.vision_language_backbone.model.score_embed,
            self.vision_language_backbone.lm_head,
        ):
            module.requires_grad_(False)

    def _build_action_modules(self):
        hidden_size = self.vision_language_backbone.config.text_config.hidden_size
        action = self.architecture["action"]
        width = self.action_hidden_size
        interaction = self.architecture["interaction"]
        self.action_dreamer = ActionDreamer(layer_num=self.action_layers, hidden_size=width,
                                           head_dim=action["head_dim"], kv_heads=action["kv_heads"])
        self.state_projector = Projector(self.state_shape[-1], width, num_layers=action["projector_layers"])
        self.action_projector = Projector(self.action_shape[-1], width, num_layers=action["projector_layers"])
        self.action_output_layer = Projector(width, self.action_shape[-1], inter_dim=width, num_layers=action["projector_layers"])
        if self.control['query_suffix']:
            self.control_mode_input = nn.Linear(1, width, bias=False)
            self.control_mode_output = nn.Linear(width, 1, bias=False)
        self.t_embedder = TimestepEmbedder(width)
        self.t_projector = Projector(width, action["time_modulations"] * width, bias=True)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            self.vision_language_backbone.config.text_config
        )
        self.sink = nn.Embedding(1, width)
        self.scene_query_base = nn.Parameter(torch.empty(1, hidden_size))
        self.scene_agent_embedding = nn.Embedding(self.max_agents, hidden_size)
        self.agent_query_projector = Projector(hidden_size, width, num_layers=action["projector_layers"])
        self.agent_query_norm = Qwen2RMSNorm(width, eps=1e-6)
        self.agent_interaction = AgentInteraction(
            hidden_size=width, num_heads=interaction["num_heads"], pose_dim=interaction["pose_dim"]
        )
        self.communication_gates = nn.Parameter(
            torch.full((self.action_dreamer.layer_num,), self.communication_gate_init)
        )
        if self.control["query_suffix"]:
            self.field_action_gates = nn.Parameter(torch.zeros(self.action_layers))
        nn.init.normal_(self.scene_query_base, mean=0.0, std=0.02)
        nn.init.normal_(self.scene_agent_embedding.weight, mean=0.0, std=0.02)

    def _build_world_modules(self):
        geometry = self.architecture["geometry"]
        self.video_expert = VideoExpert(
            config=self.architecture["video"], observed_latents=self.video_observed_latents
        )
        video_size = (
            self.video_expert.backbone.config.num_attention_heads
            * self.video_expert.backbone.config.attention_head_dim
        )
        self.ray_sampler = CameraRaySampler(
            image_size=tuple(geometry["image_size"]),
            patch_grid=tuple(geometry["patch_grid"]),
            samples_per_ray=self.field_samples_per_ray,
            bounds=self.field_bounds,
        )
        self.triplane = TriPlaneBuilder(
            input_size=video_size,
            hidden_size=self.field_size,
            resolution=self.field_resolution,
            max_observed_frames=self.video_observed_latents,
            samples_per_ray=self.field_samples_per_ray,
        )
        self.field_render_head = FieldRenderHead(
            feature_size=video_size,
            field_size=self.field_size,
            target_size=geometry["render_target_size"],
            resolution=self.field_resolution,
            max_observed_frames=self.video_observed_latents,
        )
        self.agent_field_injector = AgentFieldInjector(
            query_size=self.action_hidden_size,
            field_size=self.field_size,
            resolution=self.field_resolution,
            num_heads=self.architecture["interaction"]["num_heads"],
            num_layers=self.action_dreamer.layer_num,
            gate_init=0.01,
        )
        self.register_buffer(
            "current_field_coefficient",
            torch.tensor(self.field_coefficient_start),
            persistent=False,
        )

    @staticmethod
    def _repeat_additional_context(context, repeat):
        return {
            name: value.repeat_interleave(repeat, dim=0)
            if torch.is_tensor(value)
            else value
            for name, value in context.items()
        }

    @staticmethod
    def _slice_scene_cache(vlm_outputs, scene_length):
        if scene_length.numel() != 1:
            raise ValueError(
                "worlddreamer-v5 currently requires per-device scene batch size 1"
            )
        end = int(scene_length.item())
        past_key_values = [
            (key[:, :, :end].contiguous(), value[:, :, :end].contiguous())
            for key, value in vlm_outputs.past_key_values
        ]
        position_ids = vlm_outputs.position_ids[:, :, :end].contiguous()
        cache_mask = torch.ones(1, end, dtype=torch.bool, device=position_ids.device)
        return past_key_values, position_ids, cache_mask

    def _encode_language_context(self, batch, agent_mask, scene_length):
        input_ids = batch["input_ids"]
        query_mask = input_ids == STATE_ID
        expected = int(agent_mask.sum().item())
        if int(query_mask.sum().item()) != expected:
            raise ValueError(
                f"VLM query count {int(query_mask.sum())} does not match valid agents {expected}"
            )

        query_inputs = self.scene_query_base + self.scene_agent_embedding.weight
        query_inputs = query_inputs[None].expand(agent_mask.shape[0], -1, -1)
        state_embeds = query_inputs[agent_mask]
        vlm_outputs = self.vision_language_backbone(
            **batch,
            state_embeds=state_embeds,
            use_cache=True,
            skip_logits=True,
        )

        query_hidden = vlm_outputs.hidden_states[query_mask]
        padded = query_hidden.new_zeros(
            agent_mask.shape[0], self.max_agents, query_hidden.shape[-1]
        )
        padded[agent_mask] = query_hidden
        agent_query = self.agent_query_norm(self.agent_query_projector(padded))
        agent_query = agent_query * agent_mask.unsqueeze(-1).to(agent_query.dtype)
        return agent_query, self._slice_scene_cache(vlm_outputs, scene_length)

    @staticmethod
    def _random_mask_prefix_joint(mask, prefix_length, keep_last=2):
        if prefix_length <= keep_last:
            return mask
        action_start = 3  # sink, state, agent query
        prefix_end = action_start + prefix_length - keep_last
        suffix_start = action_start + prefix_length
        random_mask = torch.rand(prefix_length - keep_last, device=mask.device) < 0.5
        mask = mask.clone()
        mask[:, suffix_start:, action_start:prefix_end] *= (~random_mask).to(mask.dtype)
        return mask

    def _action_attention_inputs(
        self, action, action_mask, scene_position_ids, cache_mask, prefix_length
    ):
        batch_repeat, agent_count, action_length, _ = action.shape
        query_length = 1 + self.state_shape[0] + action_length + 1
        flat_batch = batch_repeat * agent_count

        source_batch = scene_position_ids.shape[1]
        repeat_factor = batch_repeat // source_batch
        position_base = scene_position_ids.max(dim=-1).values
        position_base = position_base.repeat_interleave(repeat_factor, dim=1)
        position_base = position_base.repeat_interleave(agent_count, dim=1)
        position_ids = (
            torch.arange(query_length, device=action.device).view(1, 1, -1)
            + position_base[..., None]
            + 1
        )
        if action_length > prefix_length:
            position_ids[:, :, -(action_length - prefix_length) :] += self.control["position_gap"]

        repeated_cache_mask = cache_mask.repeat_interleave(repeat_factor, dim=0)
        repeated_cache_mask = repeated_cache_mask.repeat_interleave(agent_count, dim=0)
        repeated_cache_mask = repeated_cache_mask[:, None, :].expand(
            -1, query_length, -1
        )
        causal_mask = torch.tril(
            torch.ones(
                flat_batch,
                query_length,
                query_length,
                dtype=torch.bool,
                device=action.device,
            ),
        )
        if self.control["query_suffix"]:
            causal_mask[:, self.query_index, self.action_start:self.query_index] = False
        if self.training and prefix_length > 2:
            causal_mask = self._random_mask_prefix_joint(causal_mask, prefix_length)
        attention_mask = torch.cat([repeated_cache_mask, causal_mask], dim=-1)[:, None]

        flat_action = action.flatten(0, 1)
        position_embeds = self.rotary_emb(flat_action, position_ids)
        return attention_mask, position_embeds

    @torch.no_grad()
    def _integrate_action_flow(self, noise, kwargs):
        sample = noise.clone()
        mode = kwargs.pop('control_mode_noise', None)
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            timestep = torch.full(
                (sample.shape[0], 1, 1),
                step / self.num_steps,
                device=sample.device,
                dtype=sample.dtype,
            )
            if mode is None:
                sample = sample + self.predict_action_flow(sample, timestep, **kwargs) * dt
            else:
                velocity, mode_velocity = self.predict_action_flow(sample, timestep, noisy_mode=mode, **kwargs)
                sample = (sample + velocity * dt) * kwargs['action_mask']
                mode = mode + mode_velocity * dt
        if mode is not None:
            return dict(action_latents=sample, control_mode=mode)
        return sample

    @torch.no_grad()
    def generate(self, batch):
        return self.forward(batch, return_loss=False)

    @bf16_batch
    def forward(self, batch, return_loss=False):
        batch = dict(batch)
        supplied_noise = batch.pop("inference_action_noise", None)
        supplied_mode = batch.pop('control_mode_noise', None)
        if supplied_noise is not None and (self.training or return_loss):
            raise ValueError("explicit action noise is supported only for inference")
        if self.control['query_suffix'] and (self.training or return_loss):
            raise ValueError('This converted simulation profile is inference-only')
        action = batch.pop("action")
        action_mask = batch.pop("action_mask")
        state = batch.pop("state")
        agent_mask = batch.pop("agent_mask").bool()
        relative_pose = batch.pop("relative_pose")
        scene_length = batch.pop("scene_length")
        prefix_length = (
            int(batch.pop("prefix_length", 0)) if not self.training else None
        )
        batch.pop("task", None)
        batch.pop("record_index", None)
        batch.pop("frame_index", None)
        batch.pop("labels", None)
        batch.pop("approximate_calibration_target", None)
        batch.pop("target_time_from_anchor_s", None)
        batch.pop("validation_weight", None)

        additional_context, auxiliary_losses = self._prepare_additional_context(
            batch, agent_mask
        )

        if action.ndim != 4 or state.ndim != 4:
            raise ValueError(
                f"expected joint action/state tensors, got {action.shape} and {state.shape}"
            )
        if action.shape[-2:] != self.action_shape or state.shape[-2:] != self.state_shape:
            raise ValueError(f'expected action {self.action_shape} and state {self.state_shape}, got {action.shape}, {state.shape}')
        if action_mask.shape != action.shape:
            raise ValueError('action mask must match the action tensor')
        if action.shape[1] != self.max_agents:
            raise ValueError(
                f"expected max_agents={self.max_agents}, got action shape {action.shape}"
            )

        agent_query, (past_key_values, scene_position_ids, cache_mask) = (
            self._encode_language_context(
                batch,
                agent_mask,
                scene_length,
            )
        )
        if self.training:
            prefix_length = (
                random.randint(1, 6)
                if self.async_train and random.random() < 0.5
                else 0
            )
            repeat = self.training_repeat
        else:
            repeat = 1

        if self.control["query_suffix"] and prefix_length:
            raise ValueError("RoboCasa control does not accept an action prefix")
        prefix = action[:, :, :prefix_length]
        action = action.repeat_interleave(repeat, dim=0)
        action_mask = action_mask.repeat_interleave(repeat, dim=0)
        state = state.repeat_interleave(repeat, dim=0)
        agent_mask = agent_mask.repeat_interleave(repeat, dim=0)
        relative_pose = relative_pose.repeat_interleave(repeat, dim=0)
        agent_query = agent_query.repeat_interleave(repeat, dim=0)
        prefix = prefix.repeat_interleave(repeat, dim=0)
        additional_context = self._repeat_additional_context(additional_context, repeat)

        if supplied_noise is not None:
            if supplied_noise.shape != action.shape or not torch.isfinite(supplied_noise).all():
                raise ValueError("inference noise must be finite and match the action shape")
            noise = supplied_noise.to(device=action.device, dtype=action.dtype)
        else:
            noise = torch.randn_like(action)
        kwargs = dict(
            action_mask=action_mask,
            state=state,
            agent_query=agent_query,
            agent_mask=agent_mask,
            relative_pose=relative_pose,
            scene_position_ids=scene_position_ids,
            past_key_values=past_key_values,
            cache_mask=cache_mask,
            prefix_length=prefix_length,
            **additional_context,
        )
        if not self.training:
            if self.control['query_suffix']:
                if supplied_mode is None or supplied_mode.shape != (*action.shape[:-1], 1):
                    raise ValueError('A separate control-mode noise stream is required')
                if not torch.isfinite(supplied_mode).all():
                    raise ValueError('Invalid control-mode noise')
                kwargs['control_mode_noise'] = supplied_mode.to(action)
            return self._integrate_action_flow(
                torch.cat([prefix, noise[:, :, prefix_length:]], dim=2), kwargs
            )

        timestep = (
            (1 - self.beta.sample((action.shape[0],)).to(action.device)) * 0.999
        ).to(action.dtype)
        timestep = timestep[:, None, None]
        noisy_action = (1 - timestep[:, None]) * noise + timestep[:, None] * action
        target = action - noise
        model_input = torch.cat([prefix, noisy_action[:, :, prefix_length:]], dim=2)
        pred = self.predict_action_flow(model_input, timestep, **kwargs)[
            :, :, prefix_length:
        ]
        target = target[:, :, prefix_length:]

        if prefix_length:
            generated = self._integrate_action_flow(
                torch.cat([prefix, noise[:, :, prefix_length:]], dim=2),
                kwargs,
            )
            weight = (
                generated[:, :, prefix_length:] - action[:, :, prefix_length:]
            ).abs()
        else:
            weight = torch.ones_like(pred)
        valid_mask = action_mask[:, :, prefix_length:]

        if not return_loss:
            return pred
        loss_mse, loss_freq = self.compute_joint_flow_loss(
            pred,
            target,
            valid_mask,
            weight,
            agent_mask,
        )
        loss = 0.5 * loss_mse + self.freq_coefficient * loss_freq
        return self._merge_auxiliary_losses(
            {
                "loss": loss,
                "loss_mse": loss_mse,
                "loss_freq": loss_freq,
            },
            auxiliary_losses,
        )

    def compute_joint_flow_loss(self, pred, target, action_mask, weight, agent_mask):
        pred = pred.float()
        target = target.float()
        weight = weight.float()
        mask = action_mask.bool()
        counts = mask.sum(dim=(2, 3)).clamp_min(1)
        valid_agents = agent_mask & bool_any(mask, dim=(2, 3))
        if not bool_any(valid_agents):
            zero = (pred.sum() + target.sum()) * 0.0
            return zero, zero

        with torch.no_grad():
            weight_mean = (weight * mask).sum(dim=(2, 3)) / counts
            weight = weight / weight_mean.clamp_min(1e-6)[..., None, None]
            weight.clamp_(0.5, 5.0)

        squared_error = F.mse_loss(pred, target, reduction="none") * weight
        agent_mse = (squared_error * mask).sum(dim=(2, 3)) / counts
        scene_mse = (agent_mse * valid_agents).sum(dim=1) / valid_agents.sum(
            dim=1
        ).clamp_min(1)
        loss_mse = scene_mse.mean()

        frequency_error = (
            torch.fft.rfft(pred, dim=2) - torch.fft.rfft(target, dim=2)
        ).abs()
        frequency_mask = mask[:, :, : frequency_error.shape[2]].clone()
        excluded = [
            dim for dim in self.freq_excluded_dims if dim < frequency_mask.shape[-1]
        ]
        frequency_mask[:, :, :, excluded] = False
        full_horizon = bool_any(mask[:, :, -1], dim=-1) & valid_agents
        frequency_counts = frequency_mask.sum(dim=(2, 3)).clamp_min(1)
        agent_frequency = (frequency_error * frequency_mask).sum(
            dim=(2, 3)
        ) / frequency_counts
        scene_frequency = (agent_frequency * full_horizon).sum(
            dim=1
        ) / full_horizon.sum(dim=1).clamp_min(1)
        valid_scenes = bool_any(full_horizon, dim=1)
        if bool_any(valid_scenes):
            loss_frequency = scene_frequency[valid_scenes].mean()
        else:
            loss_frequency = frequency_error.sum() * 0.0
        return loss_mse, loss_frequency

    def set_training_progress(self, progress):
        progress = min(max(float(progress), 0.0), 1.0)
        value = self.field_coefficient_start + progress * (
            self.field_coefficient_end - self.field_coefficient_start
        )
        self.current_field_coefficient.fill_(value)

    def _agent_positions(self, extrinsics, agent_mask, agent_camera_indices=None):
        if agent_camera_indices is None:
            current_wrist = extrinsics[:, 1 : self.max_agents + 1, -1].float()
        else:
            indices = agent_camera_indices
            if indices.shape != agent_mask.shape or indices.dtype != torch.long:
                raise ValueError("agent_camera_indices must be int64 [batch,agents]")
            if bool_any((indices[agent_mask] < 0) | (indices[agent_mask] >= extrinsics.shape[1])):
                raise ValueError("an active agent has no valid camera view")
            indices = indices.masked_fill(~agent_mask, 0)
            rows = torch.arange(extrinsics.shape[0], device=extrinsics.device)[:, None]
            current_wrist = extrinsics[rows, indices, -1].float()
        rotation_cw = current_wrist[..., :3].transpose(-1, -2)
        origin = -torch.einsum("...ij,...j->...i", rotation_cw, current_wrist[..., 3])
        origin = self.ray_sampler._normalize(origin).to(extrinsics.dtype)
        return origin * agent_mask.unsqueeze(-1).to(origin.dtype)

    def _prepare_additional_context(self, batch, agent_mask):
        payload = {name: batch.pop(name) for name in self.VIDEO_KEYS}
        agent_camera_indices = batch.pop("agent_camera_indices", None)
        if agent_camera_indices is not None:
            indices = agent_camera_indices
            if indices.shape != agent_mask.shape or indices.dtype != torch.long:
                raise ValueError("agent_camera_indices must be int64 [batch,agents]")
            if bool_any((indices[agent_mask] < 0) | (indices[agent_mask] >= payload["video_view_mask"].shape[1])):
                raise ValueError("an active agent has no valid camera view")
            safe_indices = indices.masked_fill(~agent_mask, 0)
            valid = payload["video_view_mask"].gather(1, safe_indices)
            roles = payload["camera_roles"].gather(1, safe_indices)
            if bool_any((~valid | (roles != 1)) & agent_mask):
                raise ValueError("agent camera indices must select visible wrist cameras")
        latent_valid = batch.pop("video_latent_valid", None)
        depth = batch.pop("camera_depth", None)
        depth_valid = batch.pop("camera_depth_valid", None)
        pixel_valid = batch.pop("camera_pixel_valid", None)
        if (depth is None) != (depth_valid is None):
            raise ValueError(
                "camera_depth and camera_depth_valid must be provided together"
            )
        if depth is None:
            leading = payload["camera_intrinsics"].shape[:-2]
            depth = payload["camera_intrinsics"].new_zeros(
                (*leading, *self.ray_sampler.patch_grid)
            )
            depth_valid = torch.zeros_like(depth, dtype=torch.bool)

        video = self.video_expert(
            payload["video_latents"],
            payload["video_text"],
            payload["video_text_mask"],
            payload["video_view_mask"],
            latent_valid,
            run_decoder=self.training,
        )
        if tuple(video["patch_grid"]) != self.ray_sampler.patch_grid:
            raise ValueError(
                f"Video patch grid {video['patch_grid']} does not match camera grid {self.ray_sampler.patch_grid}",
            )

        depth_valid = depth_valid.bool()
        if self.training and self.depth_dropout:
            keep_depth = (
                torch.rand(
                    *depth_valid.shape[:3],
                    1,
                    1,
                    device=depth_valid.device,
                )
                >= self.depth_dropout
            )
            depth_valid &= keep_depth
        points, rays, distances, sample_valid, has_depth = self.ray_sampler(
            payload["camera_intrinsics"],
            payload["camera_extrinsics"],
            depth,
            depth_valid,
        )
        points = points.flatten(3, 4)
        rays = rays.flatten(3, 4)
        distances = distances.flatten(3, 4)
        if pixel_valid is not None:
            if pixel_valid.shape != has_depth.shape:
                raise ValueError('rectified pixel validity must match the camera patch grid')
            sample_valid = sample_valid & pixel_valid.bool().unsqueeze(-1)
        sample_valid = sample_valid.flatten(3, 4)
        has_depth = has_depth.flatten(3, 4)

        features = video["scene_tokens"]
        points = points.to(features.dtype)
        rays = rays.to(features.dtype)
        distances = distances.to(features.dtype)
        workspace, _, _ = self.triplane(
            features,
            points,
            rays,
            distances,
            sample_valid,
            has_depth,
            payload["camera_roles"],
            payload["video_view_mask"],
        )

        field_loss = workspace.sum() * 0.0
        if self.training:
            reconstruction_field, _, reconstruction_weights = self.triplane(
                features.detach(),
                points,
                rays,
                distances,
                sample_valid,
                has_depth,
                payload["camera_roles"],
                payload["video_view_mask"],
            )
            prediction = self.field_render_head(
                reconstruction_field,
                points,
                rays,
                distances,
                reconstruction_weights,
                payload["camera_roles"],
            )
            target = self.field_render_head.target(features)
            token_valid = (
                bool_any(sample_valid, dim=-1) & payload["video_view_mask"][:, :, None, None]
            )
            field_loss = self.field_render_head.loss(prediction, target, token_valid)

        context = {
            "workspace": workspace,
            "agent_positions": self._agent_positions(
                payload["camera_extrinsics"], agent_mask, agent_camera_indices
            ),
        }
        losses = {
            "loss_future": video["future_loss"],
            "loss_field": field_loss,
        }
        return context, losses

    def _merge_auxiliary_losses(self, result, auxiliary_losses):
        result.update(auxiliary_losses)
        result["loss"] = (
            result["loss"]
            + self.future_coefficient * auxiliary_losses["loss_future"]
            + self.current_field_coefficient * auxiliary_losses["loss_field"]
        )
        result["field_coefficient"] = self.current_field_coefficient.detach().clone()
        return result

    def predict_action_flow(
        self,
        noisy_action,
        timestep,
        action_mask,
        state,
        agent_query,
        agent_mask,
        relative_pose,
        scene_position_ids,
        past_key_values,
        cache_mask,
        prefix_length,
        workspace,
        agent_positions,
        noisy_mode=None,
    ):
        batch_repeat, agent_count, action_length, _ = noisy_action.shape
        action_timestep = self.t_embedder(timestep[:, 0, 0] * 1000).squeeze(1)
        timestep_embed = self.t_projector(action_timestep).view(-1, self.architecture["action"]["time_modulations"], self.action_hidden_size)
        timestep_embed = timestep_embed.repeat_interleave(agent_count, dim=0)

        state_embed = self.state_projector(state)
        if noisy_mode is None:
            action_embed = self.action_projector(noisy_action * action_mask)
        else:
            action_embed = self.action_projector.layers[0](noisy_action * action_mask)
            action_embed = action_embed + self.control_mode_input(noisy_mode * agent_mask[:, :, None, None])
            for layer in self.action_projector.layers[1:]:
                action_embed = layer(action_embed)
        sink = self.sink.weight.view(1, 1, 1, -1).expand(
            batch_repeat, agent_count, -1, -1
        )
        tail = ((action_embed, agent_query.unsqueeze(2)) if self.control["query_suffix"]
                else (agent_query.unsqueeze(2), action_embed))
        hidden_states = torch.cat((sink, state_embed, *tail), dim=2).contiguous()
        action_start = self.action_start
        attention_mask, position_embeds = self._action_attention_inputs(
            noisy_action,
            action_mask,
            scene_position_ids,
            cache_mask,
            prefix_length,
        )

        layer_context = _MultiAgentActionContext(
            self,
            batch_repeat=batch_repeat,
            agent_count=agent_count,
            workspace=workspace,
            agent_positions=agent_positions,
            relative_pose=relative_pose,
            agent_mask=agent_mask,
            action_timestep=action_timestep,
        )
        hidden_states = self.action_dreamer(
            hidden_states.reshape(batch_repeat * agent_count, hidden_states.shape[2], -1),
            past_key_values,
            attention_mask,
            position_embeds,
            timestep_embed,
            layer_context=layer_context,
        ).reshape(batch_repeat, agent_count, hidden_states.shape[2], -1)

        features = hidden_states[:, :, action_start:action_start+action_length]
        if noisy_mode is not None:
            for layer in self.action_output_layer.layers[:-1]:
                features = layer(features)
            output = self.action_output_layer.layers[-1](features) * action_mask
            mode_output = self.control_mode_output(features) * agent_mask[:, :, None, None]
            return output, mode_output
        output = self.action_output_layer(features)
        if prefix_length:
            output[:, :, :prefix_length] = 0.0
        return output
