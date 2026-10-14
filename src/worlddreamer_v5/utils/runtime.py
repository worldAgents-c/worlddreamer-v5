# Copyright (C) 2026 World Agents Corporation.
import os
import torch


def configure_runtime():
    requested = os.environ.get("WORLDDREAMER_V5_SDPA_BACKEND", "auto").lower()
    if requested not in {"auto", "math", "native"}:
        raise ValueError("WORLDDREAMER_V5_SDPA_BACKEND must be auto, math, or native")
    selected = requested
    if selected == "auto":
        selected = "math" if torch.version.hip is not None else "native"
    if selected == "math":
        # The tested vendor torch can select a flash implementation whose external
        # shared library is absent from its own base image. Constrain dispatch;
        # the attention equations, masks and parameter topology are unchanged.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return selected
