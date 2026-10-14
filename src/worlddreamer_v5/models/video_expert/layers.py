# Copyright (C) 2026 World Agents Corporation.
import torch
from torch import nn
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from .attention import TokenExchangeAttention, TextReader


def _make_text_reader(enabled, width, heads, head_width, normalization):
    if not enabled:
        return None
    return TextReader(
        width,
        heads=heads,
        dim_head=head_width,
        bias=True,
        qk_norm=normalization,
        eps=1e-6,
    )


def _apply_modulated_mlp(values, normalizer, network, modulation):
    _, _, shift, scale, gate = modulation
    conditioned = normalizer(values)
    conditioned = conditioned * (1 + scale[:, None]) + shift[:, None]
    update = network(conditioned)
    return values + gate.unsqueeze(1) * update


class FusionBlock(nn.Module):
    """Update both streams after forming one normalized token sequence."""

    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio=4.0,
        qk_norm="rms_norm",
        norm_type="layer_norm",
        enable_text_cross_attention=False,
    ):
        super().__init__()
        width = num_attention_heads * attention_head_dim
        expansion = int(width * mlp_ratio)
        self.attn = TokenExchangeAttention(
            width,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            out_dim=width,
            bias=True,
            pre_only=True,
            qk_norm=qk_norm,
            eps=1e-6,
        )
        self.cross_attn = _make_text_reader(
            enable_text_cross_attention,
            width,
            num_attention_heads,
            attention_head_dim,
            qk_norm,
        )
        self.enable_text_cross_attention = enable_text_cross_attention
        self.norm = AdaLayerNormZeroSingle(width, norm_type=norm_type)
        self.proj_mlp = nn.Linear(width, expansion)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(width + expansion, width)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        attention_mask=None,
        image_rotary_emb=None,
        image_embed_seq_len=0,
    ):
        video_count = hidden_states.shape[1]
        context_count = encoder_hidden_states.shape[1]
        residual = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        normalized, gate = self.norm(residual, emb=temb)
        expansion = self.act_mlp(self.proj_mlp(normalized))
        video, context = normalized.split((video_count, context_count), dim=1)
        attended, context_update = self.attn(video, context, attention_mask, image_rotary_emb)
        if self.cross_attn is not None:
            attended = attended + self.cross_attn(
                attended, context, attention_mask, image_rotary_emb, image_embed_seq_len
            )
        merged_attention = torch.cat([attended, context_update], dim=1)
        update_input = torch.cat([merged_attention, expansion], dim=2)
        update = self.proj_out(update_input)
        result = gate.unsqueeze(1) * update + residual
        return result.split((video_count, context_count), dim=1)


class ExchangeBlock(nn.Module):
    """Keep independent stream modulation while exchanging attention outputs."""

    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio,
        qk_norm="rms_norm",
        norm_type="layer_norm",
        enable_text_cross_attention=False,
    ):
        super().__init__()
        width = num_attention_heads * attention_head_dim
        self.norm1 = AdaLayerNormZero(width, norm_type=norm_type)
        self.norm1_context = AdaLayerNormZero(width, norm_type=norm_type)
        self.attn = TokenExchangeAttention(
            width,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            added_kv_proj_dim=width,
            out_dim=width,
            bias=True,
            qk_norm=qk_norm,
            eps=1e-6,
        )
        self.cross_attn = _make_text_reader(
            enable_text_cross_attention,
            width,
            num_attention_heads,
            attention_head_dim,
            qk_norm,
        )
        self.enable_text_cross_attention = enable_text_cross_attention
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(width, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.ff_context = FeedForward(
            width, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        attention_mask=None,
        image_rotary_emb=None,
        image_embed_seq_len=0,
    ):
        video_modulation = self.norm1(hidden_states, emb=temb)
        context_modulation = self.norm1_context(encoder_hidden_states, emb=temb)
        video_update, context_update = self.attn(
            video_modulation[0],
            context_modulation[0],
            attention_mask,
            image_rotary_emb,
        )
        video = hidden_states + video_update * video_modulation[1].unsqueeze(1)
        if self.cross_attn is not None:
            video = video + self.cross_attn(
                video_update,
                context_modulation[0],
                attention_mask,
                image_rotary_emb,
                image_embed_seq_len,
            )
        context = encoder_hidden_states + context_update * context_modulation[1].unsqueeze(1)
        return (
            _apply_modulated_mlp(video, self.norm2, self.ff, video_modulation),
            _apply_modulated_mlp(
                context,
                self.norm2_context,
                self.ff_context,
                context_modulation,
            ),
        )
