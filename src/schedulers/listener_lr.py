from typing import List

import numpy as np
import torch


class ListenerLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_ratio,
        warmup_lr_ratio,
        min_factor: float = 0.0,
        max_factor: float = 100.0,
    ):
        self.total_steps = total_steps
        self.est_warmup_steps = total_steps * warmup_ratio
        self.warmup_lr_ratio = warmup_lr_ratio
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.external_factors = dict()
        super(ListenerLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.est_warmup_steps:
            lr_factor = self.warmup_lr_ratio \
                + (1 - self.warmup_lr_ratio) \
                * (self.last_epoch / self.est_warmup_steps)
        else:
            lr_factor = 1.0
        for factor in self.external_factors.values():
            lr_factor *= factor
        lr_factor = np.clip(lr_factor, self.min_factor, self.max_factor)
        lrs = [base_lr * lr_factor for base_lr in self.base_lrs]
        return lrs
