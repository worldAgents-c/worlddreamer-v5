# Copyright (C) 2026 World Agents Corporation.
from dataclasses import dataclass
from math import cos, pi
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class WarmupCosineFactor:
    """Multiplier of each parameter group's base LR, not an absolute LR.

    Post-budget cosine continues without clipping. Negative warmup start means
    constant peak during warmup. Fields can be restored by LambdaLR.load_state_dict.
    """
    warmup_updates: int
    total_updates: int
    peak: float
    initial: float
    floor: float

    def __call__(self, update: int) -> float:
        if update >= self.warmup_updates:
            progress = float(update - self.warmup_updates) / float(
                max(1, self.total_updates - self.warmup_updates))
            return (self.peak - self.floor) * 0.5 * (1.0 + cos(pi * progress)) + self.floor
        start = self.peak if self.initial < 0 else self.initial
        rise = (self.peak - start) * update / max(self.warmup_updates, 1)
        return min(self.peak, start + rise)


def warmup_cosine_scheduler(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
    max_lr: float = 4e-4, warmup_lr_start: float = 1e-6,
    min_lr: float = 8e-5, last_epoch: int = -1,
) -> LambdaLR:
    curve = WarmupCosineFactor(num_warmup_steps, num_training_steps,
                              max_lr, warmup_lr_start, min_lr)
    return LambdaLR(optimizer, curve, last_epoch=last_epoch)
