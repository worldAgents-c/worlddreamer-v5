# Copyright (C) 2026 World Agents Corporation.
import torch
from torch import nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb


def _make_head_norm(kind, width, eps, affine):
    if kind == "rms_norm":
        return nn.RMSNorm(width, eps=eps, elementwise_affine=affine)
    if kind == "layer_norm":
        return nn.LayerNorm(width, eps=eps, elementwise_affine=affine)
    return None


def _split_heads(values, projection, heads, normalization=None):
    result = projection(values).unflatten(-1, (heads, -1))
    return result if normalization is None else normalization(result)


def _rotate_prefix(values, rotation, video_length):
    if rotation is None:
        return values
    rotated = apply_rotary_emb(values[:, :video_length], rotation, sequence_dim=1)
    return (
        rotated
        if video_length == values.shape[1]
        else torch.cat((rotated, values[:, video_length:]), dim=1)
    )


def _run_attention(query, key, value, mask):
    merged = dispatch_attention_fn(
        query, key, value, attn_mask=mask, backend=None, parallel_config=None
    )
    return merged.flatten(2, 3).to(query.dtype)


def _apply_output(stack, values):
    if stack is None:
        return values
    values = stack[0](values)
    return stack[1](values)


class TextReader(nn.Module):
    """Read the text suffix with queries produced from the video sequence."""

    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        bias=False,
        out_bias=True,
        eps=1e-5,
        qk_norm="rms_norm",
        elementwise_affine=True,
    ):
        super().__init__()
        self.heads = heads
        width = heads * dim_head
        self.to_q = nn.Linear(query_dim, width, bias=bias)
        self.to_k = nn.Linear(query_dim, width, bias=bias)
        self.to_v = nn.Linear(query_dim, width, bias=bias)
        self.norm_q = _make_head_norm(qk_norm, dim_head, eps, elementwise_affine)
        self.norm_k = _make_head_norm(qk_norm, dim_head, eps, elementwise_affine)
        self.to_out = nn.ModuleList(
            [nn.Linear(width, query_dim, bias=out_bias), nn.Dropout(dropout)]
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
        image_embed_seq_len=0,
    ):
        text_tokens = encoder_hidden_states[:, image_embed_seq_len:]
        text_count = text_tokens.shape[1]
        if text_count == 0:
            raise ValueError("text attention requires at least one text token")
        mask = (
            None
            if attention_mask is None
            else attention_mask[:, :, : hidden_states.shape[1], -text_count:]
        )
        query = _split_heads(hidden_states, self.to_q, self.heads, self.norm_q)
        key = _split_heads(text_tokens, self.to_k, self.heads, self.norm_k)
        value = _split_heads(text_tokens, self.to_v, self.heads)
        query = _rotate_prefix(query, image_rotary_emb, hidden_states.shape[1])
        return _apply_output(self.to_out, _run_attention(query, key, value, mask))


class TokenExchangeAttention(nn.Module):
    """Fuse video tokens with an optional second token stream."""

    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        bias=False,
        added_kv_proj_dim=None,
        added_proj_bias=True,
        out_bias=True,
        eps=1e-5,
        out_dim=None,
        elementwise_affine=True,
        pre_only=False,
        context_pre_only=False,
        qk_norm="rms_norm",
    ):
        super().__init__()
        width = dim_head * heads if out_dim is None else out_dim
        self.heads = width // dim_head
        self.to_q = nn.Linear(query_dim, width, bias=bias)
        self.to_k = nn.Linear(query_dim, width, bias=bias)
        self.to_v = nn.Linear(query_dim, width, bias=bias)
        self.norm_q = _make_head_norm(qk_norm, dim_head, eps, elementwise_affine)
        self.norm_k = _make_head_norm(qk_norm, dim_head, eps, elementwise_affine)
        self.to_out = (
            None
            if pre_only
            else nn.ModuleList(
                [
                    nn.Linear(
                        width, query_dim if out_dim is None else out_dim, bias=out_bias
                    ),
                    nn.Dropout(dropout),
                ]
            )
        )
        self.norm_added_q = self.norm_added_k = None
        self.add_q_proj = self.add_k_proj = self.add_v_proj = self.to_add_out = None
        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = nn.Linear(
                added_kv_proj_dim, width, bias=added_proj_bias
            )
            self.add_k_proj = nn.Linear(
                added_kv_proj_dim, width, bias=added_proj_bias
            )
            self.add_v_proj = nn.Linear(
                added_kv_proj_dim, width, bias=added_proj_bias
            )
            if not context_pre_only:
                self.to_add_out = nn.Linear(width, query_dim, bias=out_bias)

    def video_projections(self, values, rotation, length):
        query = _split_heads(values, self.to_q, self.heads, self.norm_q)
        key = _split_heads(values, self.to_k, self.heads, self.norm_k)
        value = _split_heads(values, self.to_v, self.heads)
        return (
            _rotate_prefix(query, rotation, length),
            _rotate_prefix(key, rotation, length),
            value,
        )

    def context_projections(self, values):
        return (
            _split_heads(values, self.add_q_proj, self.heads, self.norm_added_q),
            _split_heads(values, self.add_k_proj, self.heads, self.norm_added_k),
            _split_heads(values, self.add_v_proj, self.heads),
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        context = encoder_hidden_states
        video_length = hidden_states.shape[1]
        shared = context is not None and self.add_q_proj is None
        source = torch.cat((hidden_states, context), dim=1) if shared else hidden_states
        projected = self.video_projections(source, image_rotary_emb, video_length)
        if context is not None and not shared:
            projected = tuple(
                torch.cat(pair, dim=1)
                for pair in zip(projected, self.context_projections(context))
            )
        output = _run_attention(*projected, attention_mask)
        video = output[:, :video_length]
        video = _apply_output(self.to_out, video)
        if context is None:
            return video
        context = output[:, video_length:]
        if self.to_add_out is not None:
            context = self.to_add_out(context)
        return video, context
