# Copyright (C) 2026 World Agents Corporation.
from functools import wraps
from typing import Callable
import torch


def cast_batch_floats(batch: dict, *, source=torch.float32, target=torch.bfloat16):
    """Update top-level matching tensors in place, retaining device and autograd.

    Nested payloads and integer masks retain their original semantics.
    """
    fields = tuple(name for name, tensor in batch.items()
                   if isinstance(tensor, torch.Tensor) and tensor.dtype == source)
    for name in fields:
        batch[name] = batch[name].to(dtype=target)
    return batch


def bf16_batch(method: Callable) -> Callable:
    """Apply the batch precision policy at the model call boundary."""
    @wraps(method)
    def invoke(model, batch, *args, **kwargs):
        return method(model, cast_batch_floats(batch), *args, **kwargs)
    return invoke
