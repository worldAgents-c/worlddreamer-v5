# Copyright (C) 2026 World Agents Corporation.
import os
import torch

__all__ = ["KERNEL_BACKEND", "LigerRMSNorm", "liger_rotary_pos_emb"]

KERNEL_BACKEND = os.environ.get("WORLDDREAMER_V5_KERNEL_BACKEND", "torch").lower()
if KERNEL_BACKEND not in {"auto", "torch", "liger"}:
    raise ValueError("WORLDDREAMER_V5_KERNEL_BACKEND must be auto, torch, or liger")
if KERNEL_BACKEND == "auto":
    KERNEL_BACKEND = (
        "liger" if torch.cuda.is_available() and torch.version.hip is None else "torch"
    )

if KERNEL_BACKEND == "liger":
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
else:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as LigerRMSNorm
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb as liger_rotary_pos_emb,
    )
