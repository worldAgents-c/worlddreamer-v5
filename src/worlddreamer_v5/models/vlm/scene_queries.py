# Copyright (C) 2026 World Agents Corporation.
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class AgentInteraction(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8, pose_dim=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pose_bias = nn.Sequential(
            nn.Linear(pose_dim, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, num_heads, bias=True),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, queries, relative_pose, agent_mask):
        batch_size, agent_count, _ = queries.shape
        qkv = self.qkv_proj(self.norm(queries)).view(
            batch_size,
            agent_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        bias = self.pose_bias(relative_pose).permute(0, 3, 1, 2).to(query.dtype)
        key_mask = agent_mask[:, None, None, :]
        bias = bias.masked_fill(~key_mask, torch.finfo(query.dtype).min)
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0
        )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, agent_count, self.hidden_size)
        )
        output = self.out_proj(output)
        return output * agent_mask.unsqueeze(-1).to(output.dtype)
