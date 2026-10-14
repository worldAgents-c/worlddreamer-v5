# Copyright (C) 2026 World Agents Corporation.
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half


def initialize_projection(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Projector(nn.Module):
    """Stacked affine maps; keep the state encoder's parameter layout intact."""
    def __init__(self, input_dim, output_dim, inter_dim=None, num_layers=1, bias=False):
        super().__init__()
        if num_layers < 1:
            raise ValueError('a projection must contain at least one linear layer')
        middle = output_dim if inter_dim is None else inter_dim
        widths = [input_dim] + [middle] * (num_layers - 1) + [output_dim]
        modules = []
        for index, (incoming, outgoing) in enumerate(zip(widths, widths[1:])):
            if index:
                modules.append(nn.GELU(approximate='tanh'))
            modules.append(nn.Linear(incoming, outgoing, bias=bias))
        self.layers = nn.Sequential(*modules)
        self.apply(initialize_projection)

    def forward(self, values):
        return self.layers(values)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.encoder = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=False))

    def harmonic_features(self, times):
        bands = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            bands, dtype=torch.float32, device=times.device) / bands)
        phases = times[:, None].float() * frequencies[None]
        return torch.cat((phases.cos(), phases.sin()), dim=-1).to(torch.bfloat16)

    def forward(self, times):
        return self.encoder(self.harmonic_features(times))[:, None]


def align_condition_batch(tensor, batch):
    """Repeat each scene cache for its actions, preserving scene-major ordering."""
    if tensor.shape[0] == batch:
        return tensor
    if batch % tensor.shape[0]:
        raise ValueError(f'condition batch {tensor.shape[0]} cannot expand to {batch}')
    return tensor.repeat_interleave(batch // tensor.shape[0], dim=0)


def rotate_queries_and_keys(query, key, rotation):
    cosine, sine = rotation
    if cosine.ndim == 4:
        cosine, sine = cosine[0], sine[0]
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    return (query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine)


class ConditionedAttention(nn.Module):
    def __init__(self, hidden_size=1024, head_dim=128, kv_heads=8):
        super().__init__()
        self.width, self.head_width = hidden_size, head_dim
        self.heads = hidden_size // head_dim
        self.cache_repeats = self.heads // kv_heads
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_norm = Qwen2RMSNorm(head_dim)
        self.key_norm = Qwen2RMSNorm(head_dim)

    def project(self, tokens, rotation):
        batch, length, _ = tokens.shape
        packed = self.query_key_value(tokens).view(batch, length, 3, self.heads, self.head_width)
        query, key, value = packed.unbind(2)
        query = self.query_norm(query).transpose(1, 2)
        key = self.key_norm(key).transpose(1, 2)
        query, key = rotate_queries_and_keys(query, key, rotation)
        return query, key, value.transpose(1, 2)

    def condition_prefix(self, cache, batch):
        return tuple(align_condition_batch(item, batch).repeat_interleave(
            self.cache_repeats, dim=1) for item in cache)

    def forward(self, tokens, cache, rotation, mask):
        query, key, value = self.project(tokens, rotation)
        prefix_key, prefix_value = self.condition_prefix(cache, tokens.shape[0])
        context = F.scaled_dot_product_attention(
            query, torch.cat((prefix_key, key), dim=-2),
            torch.cat((prefix_value, value), dim=-2), attn_mask=mask, dropout_p=0.0)
        merged = context.transpose(1, 2).contiguous().view(*tokens.shape[:2], self.width)
        return self.output(merged)


class GatedFeedForward(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        expansion = 4 * hidden_size
        self.gate = nn.Linear(hidden_size, expansion, bias=False)
        self.value = nn.Linear(hidden_size, expansion, bias=False)
        self.output = nn.Linear(expansion, hidden_size, bias=False)

    def forward(self, tokens):
        return self.output(F.silu(self.gate(tokens)) * self.value(tokens))


class ActionBlock(nn.Module):
    def __init__(self, hidden_size=1024, head_dim=128, kv_heads=8):
        super().__init__()
        self.attention = ConditionedAttention(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads)
        self.feed_forward = GatedFeedForward(hidden_size)
        self.attention_norm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward_norm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.modulation = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    @staticmethod
    def condition(tokens, norm, shift, scale):
        return norm(tokens) * (1 + scale) + shift

    def forward(self, tokens, cache, rotation, time_condition, mask):
        attention_shift, attention_scale, attention_gate, mlp_shift, mlp_scale, mlp_gate = (
            self.modulation[None] + time_condition).chunk(6, dim=1)
        attention_input = self.condition(tokens, self.attention_norm, attention_shift, attention_scale)
        tokens = tokens + attention_gate * self.attention(attention_input, cache, rotation, mask)
        feed_forward_input = self.condition(tokens, self.feed_forward_norm, mlp_shift, mlp_scale)
        return tokens + mlp_gate * self.feed_forward(feed_forward_input)


class ActionLayerContext:
    """Optional per-layer adapter for action-stack conditioning.

    The action stack owns layer traversal. A context can adapt the token
    layout before and after each block without adding parameters or changing
    the action block state-dict layout.
    """

    def before_block(self, tokens, layer_index):
        return tokens

    def after_block(self, tokens, layer_index):
        return tokens


class ActionDreamer(nn.Module):
    def __init__(self, layer_num=36, hidden_size=1024, head_dim=128, kv_heads=8):
        super().__init__()
        self.layer_num = layer_num
        self.blocks = nn.ModuleList(ActionBlock(hidden_size, head_dim, kv_heads) for _ in range(layer_num))
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        initialize_projection(module)
        if isinstance(module, Qwen2RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        tokens,
        past_key_values,
        attn_mask,
        position_embeds,
        timestep,
        layer_context=None,
    ):
        offset = len(past_key_values) - self.layer_num
        if offset < 0:
            raise ValueError('not enough condition layers for the action stack')
        for index, block in enumerate(self.blocks):
            if layer_context is not None:
                tokens = layer_context.before_block(tokens, index)
            tokens = block(tokens, past_key_values[offset + index], position_embeds,
                           timestep, attn_mask)
            if layer_context is not None:
                tokens = layer_context.after_block(tokens, index)
        return tokens
