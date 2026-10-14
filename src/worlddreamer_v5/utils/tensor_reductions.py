# Copyright (C) 2026 World Agents Corporation.
import torch


def bool_any(value, dim=None, keepdim=False):
    if value.device.type == "cuda" and torch.version.hip is not None:
        return value.bool().sum(dim=dim, keepdim=keepdim, dtype=torch.int64) != 0
    return torch.any(value, dim=dim, keepdim=keepdim)


def bool_all(value, dim=None, keepdim=False):
    if value.device.type == "cuda" and torch.version.hip is not None:
        return value.logical_not().sum(dim=dim, keepdim=keepdim, dtype=torch.int64) == 0
    return torch.all(value, dim=dim, keepdim=keepdim)
