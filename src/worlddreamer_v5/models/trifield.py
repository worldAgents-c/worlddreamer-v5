# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
from worlddreamer_v5.utils.tensor_reductions import bool_any, bool_all

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


PLANE_AXES = ((0, 1), (0, 2), (1, 2))  # XY, XZ, YZ


def prefix_asymmetric_mask(observed_tokens, future_tokens, text_mask):
    """Build a leak-free [video, text] attention mask for video encoder blocks."""
    observed_tokens = int(observed_tokens)
    future_tokens = int(future_tokens)
    if observed_tokens <= 0 or future_tokens < 0 or text_mask.ndim != 2:
        raise ValueError("invalid prefix mask dimensions")

    batch_size, text_tokens = text_mask.shape
    video_tokens = observed_tokens + future_tokens
    total_tokens = video_tokens + text_tokens
    mask = torch.zeros(
        batch_size,
        total_tokens,
        total_tokens,
        dtype=torch.bool,
        device=text_mask.device,
    )
    mask[:, :observed_tokens, :observed_tokens] = True
    mask[:, observed_tokens:video_tokens, :video_tokens] = True
    mask[:, video_tokens:, :observed_tokens] = True
    mask[:, :, video_tokens:] = text_mask[:, None, :].bool()
    return mask[:, None]


class AxisMLP(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(length, length * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(length * 2, length),
        )

    def forward(self, x):
        return x + self.net(x)


class TriPlaneAxisMixer(nn.Module):
    """Mix within and across planes without pooling or dense token attention."""

    def __init__(self, resolution=16):
        super().__init__()
        self.row = AxisMLP(resolution)
        self.column = AxisMLP(resolution)
        self.shared_x = AxisMLP(resolution * 2)
        self.shared_y = AxisMLP(resolution * 2)
        self.shared_z = AxisMLP(resolution * 2)

    @staticmethod
    def _mix_pair(first, second, mixer):
        joined = torch.cat([first, second], dim=-1)
        first, second = mixer(joined).chunk(2, dim=-1)
        return first, second

    def forward(self, field):
        if field.ndim != 5 or field.shape[1] != 3 or field.shape[2] != field.shape[3]:
            raise ValueError(
                f"expected [B,3,R,R,C] Tri-plane, got {tuple(field.shape)}"
            )

        # MLPs operate on spatial axes independently for every channel.
        x = field.permute(0, 1, 4, 2, 3)
        x = self.row(x)
        x = self.column(x.transpose(-1, -2)).transpose(-1, -2)
        xy, xz, yz = x.unbind(1)

        # X: rows XY[x, y] <-> XZ[x, z].
        xy, xz = self._mix_pair(xy, xz, self.shared_x)
        # Y: columns XY[x, y] <-> rows YZ[y, z].
        xy_y, yz = self._mix_pair(xy.transpose(-1, -2), yz, self.shared_y)
        xy = xy_y.transpose(-1, -2)
        # Z: columns XZ[x, z] <-> columns YZ[y, z].
        xz_z, yz_z = self._mix_pair(
            xz.transpose(-1, -2),
            yz.transpose(-1, -2),
            self.shared_z,
        )
        xz = xz_z.transpose(-1, -2)
        yz = yz_z.transpose(-1, -2)
        return torch.stack([xy, xz, yz], dim=1).permute(0, 1, 3, 4, 2)


class WorkspaceFilter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.old_norm = nn.LayerNorm(hidden_size)
        self.new_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 2.0)

    def forward(self, old, new, visibility, delta_t):
        delta_t = torch.as_tensor(delta_t, device=new.device, dtype=new.dtype)
        delta_t = delta_t.expand(*new.shape[:-1], 1)
        gate_input = torch.cat(
            [self.old_norm(old), self.new_norm(new), visibility.to(new.dtype), delta_t],
            dim=-1,
        )
        # A cell with no supporting ray is unknown, not an observation of zero.
        # Preserve its previous state instead of erasing it toward the empty
        # splat buffer.
        gate = torch.sigmoid(self.gate(gate_input)) * visibility.to(new.dtype)
        return old + gate * (new - old)


class CameraRaySampler(nn.Module):
    """Back-project metric depth or produce sparse inverse-depth AABB samples."""

    def __init__(
        self, image_size=(240, 320), patch_grid=(15, 20), samples_per_ray=8, bounds=None
    ):
        super().__init__()
        self.image_size = tuple(int(value) for value in image_size)
        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.samples_per_ray = int(samples_per_ray)
        if bounds is None:
            bounds = ((-0.8, -1.0, -0.05), (1.2, 1.0, 1.2))
        bounds = torch.tensor(bounds, dtype=torch.float32)
        self.register_buffer("bounds_min", bounds[0], persistent=True)
        self.register_buffer("bounds_max", bounds[1], persistent=True)

        height, width = self.image_size
        grid_h, grid_w = self.patch_grid
        pixel_y = (
            torch.arange(grid_h, dtype=torch.float32) + 0.5
        ) * height / grid_h - 0.5
        pixel_x = (
            torch.arange(grid_w, dtype=torch.float32) + 0.5
        ) * width / grid_w - 0.5
        yy, xx = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        self.register_buffer("pixels", pixels, persistent=False)

    def _normalize(self, points):
        scale = (self.bounds_max - self.bounds_min).clamp_min(1e-6)
        return 2.0 * (points - self.bounds_min) / scale - 1.0

    def forward(self, intrinsics, extrinsics, depth, depth_valid):
        if intrinsics.shape[:-2] != extrinsics.shape[:-2]:
            raise ValueError(
                "intrinsics and extrinsics must have the same leading dimensions"
            )
        if intrinsics.shape[-2:] != (3, 3) or extrinsics.shape[-2:] != (3, 4):
            raise ValueError("camera matrices must be [...,3,3] and [...,3,4]")

        leading = intrinsics.shape[:-2]
        grid_h, grid_w = self.patch_grid
        if depth.shape != (*leading, grid_h, grid_w):
            raise ValueError(
                f"expected depth {(*leading, grid_h, grid_w)}, got {tuple(depth.shape)}"
            )

        pixels = self.pixels.to(intrinsics).expand(*leading, grid_h, grid_w, 3)
        camera_dirs = torch.einsum(
            "...ij,...hwj->...hwi", torch.linalg.inv(intrinsics.float()), pixels.float()
        )
        rotation_wc = extrinsics[..., :3].float()
        translation_wc = extrinsics[..., 3].float()
        rotation_cw = rotation_wc.transpose(-1, -2)
        origins = -torch.einsum("...ij,...j->...i", rotation_cw, translation_wc)
        world_dirs_z = torch.einsum("...ij,...hwj->...hwi", rotation_cw, camera_dirs)
        world_dirs = F.normalize(world_dirs_z, dim=-1)

        safe_dirs = torch.where(
            world_dirs.abs() < 1e-6,
            world_dirs.sign() * 1e-6 + (world_dirs == 0) * 1e-6,
            world_dirs,
        )
        bounds_min = self.bounds_min.float()
        bounds_max = self.bounds_max.float()
        t0 = (bounds_min - origins[..., None, None, :]) / safe_dirs
        t1 = (bounds_max - origins[..., None, None, :]) / safe_dirs
        near = torch.minimum(t0, t1).amax(dim=-1).clamp_min(1e-3)
        far = torch.maximum(t0, t1).amin(dim=-1)
        ray_hits = far > near

        alpha = torch.linspace(0.0, 1.0, self.samples_per_ray, device=depth.device)
        inverse = (1.0 - alpha) / near[..., None] + alpha / far.clamp_min(1e-3)[
            ..., None
        ]
        fallback_distance = inverse.clamp_min(1e-6).reciprocal()
        fallback_points = (
            origins[..., None, None, None, :]
            + world_dirs[..., None, :] * fallback_distance[..., None]
        )

        depth = depth.float()
        depth_points_camera = camera_dirs * depth[..., None]
        depth_points_world = (
            torch.einsum("...ij,...hwj->...hwi", rotation_cw, depth_points_camera)
            + origins[..., None, None, :]
        )
        depth_distance = torch.linalg.norm(
            depth_points_world - origins[..., None, None, :], dim=-1
        )
        depth_points = depth_points_world[..., None, :].expand_as(fallback_points)
        depth_distances = depth_distance[..., None].expand_as(fallback_distance)
        use_depth = depth_valid.bool() & torch.isfinite(depth) & (depth > 0)

        points = torch.where(use_depth[..., None, None], depth_points, fallback_points)
        distances = torch.where(
            use_depth[..., None], depth_distances, fallback_distance
        )
        valid = torch.where(
            use_depth[..., None],
            torch.ones_like(fallback_distance, dtype=torch.bool),
            ray_hits[..., None],
        )
        normalized = self._normalize(points)
        valid &= bool_all(normalized.abs().le(1.0), dim=-1)
        return (
            normalized.to(intrinsics.dtype),
            world_dirs.to(intrinsics.dtype),
            distances.to(intrinsics.dtype),
            valid,
            use_depth,
        )


class TriPlaneBuilder(nn.Module):
    def __init__(
        self,
        input_size=1536,
        hidden_size=256,
        resolution=16,
        max_observed_frames=3,
        samples_per_ray=8,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.resolution = int(resolution)
        self.samples_per_ray = int(samples_per_ray)
        # Video hidden states depend on the sampled flow timestep.  Normalize
        # them before geometry fusion so that this nuisance scale does not leak
        # into the workspace confidence or its recurrent update.
        self.feature_norm = nn.LayerNorm(input_size)
        self.feature_proj = nn.Linear(input_size, hidden_size)
        self.camera_role_embedding = nn.Embedding(2, hidden_size)
        self.time_embedding = nn.Embedding(max_observed_frames, hidden_size)
        self.geometry = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ray_confidence = nn.Sequential(
            nn.Linear(hidden_size + 8, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.axis_mixer = TriPlaneAxisMixer(resolution)
        self.filter = WorkspaceFilter(hidden_size)
        self.register_buffer(
            "zero_workspace",
            torch.zeros(1, 3, resolution, resolution, hidden_size),
            persistent=False,
        )

    def _splat(self, features, points, weights, valid):
        batch_size, count, hidden_size = features.shape
        accumulation_dtype = (
            torch.float32 if features.dtype in (torch.float16, torch.bfloat16)
            else features.dtype
        )
        output = features.new_zeros(
            batch_size, 3, self.resolution * self.resolution, hidden_size,
            dtype=accumulation_dtype,
        )
        density = features.new_zeros(
            batch_size, 3, self.resolution * self.resolution, 1,
            dtype=accumulation_dtype,
        )
        for plane, axes in enumerate(PLANE_AXES):
            coordinates = (points[..., list(axes)] + 1.0) * 0.5 * (self.resolution - 1)
            lower = coordinates.floor().long().clamp(0, self.resolution - 1)
            upper = (lower + 1).clamp(0, self.resolution - 1)
            fraction = coordinates - lower.to(coordinates.dtype)
            for x_index, x_weight in (
                (lower[..., 0], 1 - fraction[..., 0]),
                (upper[..., 0], fraction[..., 0]),
            ):
                for y_index, y_weight in (
                    (lower[..., 1], 1 - fraction[..., 1]),
                    (upper[..., 1], fraction[..., 1]),
                ):
                    index = x_index * self.resolution + y_index
                    contribution = (weights * valid * x_weight * y_weight).unsqueeze(-1)
                    output[:, plane].scatter_add_(
                        1,
                        index.unsqueeze(-1).expand(-1, -1, hidden_size),
                        (features * contribution).to(accumulation_dtype),
                    )
                    density[:, plane].scatter_add_(
                        1, index.unsqueeze(-1), contribution.to(accumulation_dtype)
                    )
        denominator = density.clamp_min(1e-6).masked_fill(density == 0, 1.0)
        output = (output / denominator).to(features.dtype)
        density = density.to(features.dtype)
        return (
            output.view(batch_size, 3, self.resolution, self.resolution, hidden_size),
            density.view(batch_size, 3, self.resolution, self.resolution, 1),
        )

    def forward(
        self,
        features,
        points,
        ray_directions,
        distances,
        valid,
        has_depth,
        camera_roles,
        view_mask,
    ):
        batch_size, view_count, time_count, patch_count, input_size = features.shape
        if points.shape[:4] != features.shape[:4] or points.shape[-2:] != (
            self.samples_per_ray,
            3,
        ):
            raise ValueError("feature and ray sample shapes do not match")

        base = self.feature_proj(self.feature_norm(features))
        base = base + self.camera_role_embedding(camera_roles)[:, :, None, None]
        workspace = self.zero_workspace.to(base).expand(batch_size, -1, -1, -1, -1)
        visibility_history = []
        weight_history = []
        for time_index in range(time_count):
            time_base = base[:, :, time_index] + self.time_embedding.weight[time_index]
            time_points = points[:, :, time_index]
            time_rays = ray_directions[:, :, time_index, :, None].expand_as(time_points)
            time_distances = distances[:, :, time_index, :, :, None]
            depth_flag = has_depth[:, :, time_index, :, None, None].expand(
                *time_points.shape[:-1], 1
            )
            geometry = torch.cat(
                [
                    time_points,
                    time_rays,
                    time_distances,
                    depth_flag.to(time_points.dtype),
                ],
                dim=-1,
            )
            expanded = time_base[..., None, :].expand(
                *geometry.shape[:-1], self.hidden_size
            )
            sample_features = expanded + self.geometry(geometry)
            logits = self.ray_confidence(
                torch.cat([expanded, geometry], dim=-1)
            ).squeeze(-1)
            sample_valid = valid[:, :, time_index] & view_mask[:, :, None, None]
            logits = logits.masked_fill(~sample_valid, -1e4)
            learned_weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
            depth_weights = torch.zeros_like(learned_weights)
            depth_weights[..., 0] = 1.0
            sample_weights = torch.where(
                has_depth[:, :, time_index, :, None], depth_weights, learned_weights
            )
            sample_weights = sample_weights * sample_valid

            flat_features = sample_features.flatten(1, 3)
            flat_points = time_points.flatten(1, 3)
            flat_weights = sample_weights.flatten(1, 3)
            flat_valid = sample_valid.flatten(1, 3)
            raw, visibility = self._splat(
                flat_features, flat_points, flat_weights, flat_valid
            )
            raw = self.axis_mixer(raw)
            visibility = visibility.clamp(0, 1)
            workspace = self.filter(workspace, raw, visibility, delta_t=1.0)
            visibility_history.append(visibility)
            weight_history.append(sample_weights)
        return (
            workspace,
            torch.stack(visibility_history, dim=1),
            torch.stack(weight_history, dim=2),
        )


class AgentFieldInjector(nn.Module):
    def __init__(
        self,
        query_size=1024,
        field_size=256,
        resolution=16,
        num_heads=8,
        num_layers=36,
        gate_init=0.01,
    ):
        super().__init__()
        self.query_size = int(query_size)
        self.field_size = int(field_size)
        self.resolution = int(resolution)
        self.num_heads = int(num_heads)
        self.head_dim = query_size // num_heads
        self.query_norm = Qwen2RMSNorm(query_size, eps=1e-6)
        self.field_norm = nn.LayerNorm(field_size)
        self.query_proj = nn.Linear(query_size, query_size, bias=True)
        self.key_proj = nn.Linear(field_size, query_size, bias=True)
        self.value_proj = nn.Linear(field_size, query_size, bias=True)
        self.out_proj = nn.Linear(query_size, query_size, bias=False)
        self.distance_scale = nn.Parameter(torch.zeros(num_heads))
        self.modulator = nn.Sequential(
            nn.Linear(query_size * 2, query_size),
            nn.SiLU(),
            nn.Linear(query_size, query_size * 3),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.modulator[-1].bias.data[query_size * 2 :].fill_(gate_logit)
        self.layer_modulation = nn.Parameter(torch.zeros(num_layers, query_size * 3))

        coordinates = torch.linspace(-1.0, 1.0, resolution)
        first, second = torch.meshgrid(coordinates, coordinates, indexing="ij")
        plane_coordinates = []
        for axes in PLANE_AXES:
            xyz = torch.zeros(resolution, resolution, 3)
            xyz[..., axes[0]] = first
            xyz[..., axes[1]] = second
            plane_coordinates.append(xyz)
        self.register_buffer(
            "plane_coordinates",
            torch.stack(plane_coordinates).flatten(0, 2),
            persistent=False,
        )

    def forward(
        self,
        queries,
        field,
        agent_positions,
        agent_mask,
        action_timestep,
        layer_index=0,
    ):
        batch_size, agent_count, _ = queries.shape
        tokens = self.field_norm(field).flatten(1, 3)
        query = self.query_proj(self.query_norm(queries)).view(
            batch_size, agent_count, self.num_heads, self.head_dim
        )
        key = self.key_proj(tokens).view(batch_size, -1, self.num_heads, self.head_dim)
        value = self.value_proj(tokens).view(
            batch_size, -1, self.num_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        field_coordinates = self.plane_coordinates.to(agent_positions)
        distances = []
        tokens_per_plane = self.resolution * self.resolution
        for plane, axes in enumerate(PLANE_AXES):
            token_slice = slice(
                plane * tokens_per_plane, (plane + 1) * tokens_per_plane
            )
            delta = (
                agent_positions[..., None, list(axes)]
                - field_coordinates[token_slice, list(axes)]
            )
            distances.append(torch.linalg.norm(delta.float(), dim=-1))
        distance = torch.cat(distances, dim=-1).to(query.dtype)
        scale = (
            F.softplus(self.distance_scale)
            .view(1, self.num_heads, 1, 1)
            .to(query.dtype)
        )
        attention_bias = -scale * distance[:, None]
        attention_bias = attention_bias.masked_fill(~agent_mask[:, None, :, None], -1e4)
        residual = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_bias, dropout_p=0.0
        )
        residual = self.out_proj(
            residual.transpose(1, 2).reshape(batch_size, agent_count, self.query_size)
        )

        action_timestep = action_timestep[:, None].expand(-1, agent_count, -1)
        modulation = self.modulator(torch.cat([queries, action_timestep], dim=-1))
        modulation = modulation + self.layer_modulation[layer_index]
        scale, bias, gate = modulation.chunk(3, dim=-1)
        update = torch.sigmoid(gate) * ((1.0 + scale) * residual + bias)
        return (queries + update) * agent_mask.unsqueeze(-1).to(queries.dtype)


class FieldRenderHead(nn.Module):
    def __init__(
        self,
        feature_size=1536,
        field_size=256,
        target_size=256,
        resolution=16,
        max_observed_frames=3,
    ):
        super().__init__()
        self.field_size = int(field_size)
        self.target_size = int(target_size)
        self.resolution = int(resolution)
        self.camera_role_embedding = nn.Embedding(2, 32)
        self.time_embedding = nn.Embedding(max_observed_frames, 32)
        self.decoder = nn.Sequential(
            nn.Linear(field_size * 3 + 3 + 1 + 1 + 32 + 32, field_size),
            nn.SiLU(),
            nn.Linear(field_size, target_size),
        )
        generator = torch.Generator(device="cpu").manual_seed(20260902)
        projection = torch.randn(
            feature_size, target_size, generator=generator
        ) / math.sqrt(feature_size)
        self.register_buffer("target_projection", projection, persistent=True)

    def _sample_planes(self, field, points):
        batch_size = field.shape[0]
        samples = []
        for plane, axes in enumerate(PLANE_AXES):
            grid = points[..., list(reversed(axes))].reshape(batch_size, -1, 1, 2)
            plane_field = field[:, plane].permute(0, 3, 1, 2)
            sampled = F.grid_sample(
                plane_field, grid, mode="bilinear", align_corners=True
            )
            samples.append(
                sampled.squeeze(-1)
                .transpose(1, 2)
                .reshape(*points.shape[:-1], self.field_size)
            )
        return torch.cat(samples, dim=-1)

    def forward(self, field, points, ray_directions, distances, weights, camera_roles):
        batch_size, view_count, time_count, patch_count, sample_count, _ = points.shape
        sampled = self._sample_planes(field, points)
        rays = ray_directions[..., None, :].expand_as(points)
        role = self.camera_role_embedding(camera_roles)[:, :, None, None, None].expand(
            batch_size, view_count, time_count, patch_count, sample_count, -1
        )
        time = self.time_embedding.weight[:time_count][
            None, None, :, None, None
        ].expand(batch_size, view_count, time_count, patch_count, sample_count, -1)
        confidence = weights[..., None]
        decoded = self.decoder(
            torch.cat(
                [sampled, rays, distances[..., None], confidence, role, time], dim=-1
            )
        )
        return (decoded * confidence).sum(dim=-2)

    def target(self, clean_features):
        clean_features = F.layer_norm(
            clean_features.float(), (clean_features.shape[-1],)
        )
        return (
            clean_features.matmul(self.target_projection.float())
            .to(clean_features.dtype)
            .detach()
        )

    @staticmethod
    def loss(prediction, target, valid):
        valid = valid.bool()
        if not bool_any(valid):
            return prediction.sum() * 0.0
        prediction = prediction[valid].float()
        target = target[valid].float()
        cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
        huber = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        return (cosine + 0.1 * huber).mean()
