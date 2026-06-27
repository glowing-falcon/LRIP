from typing import List

import numpy as np
import torch


class CyclicStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_lr_ratio, target_lr_ratio, total_steps, warmup_ratio, cycles=1):
        self.total_steps = total_steps
        self.cycles = cycles
        self.steps_per_cycle = self.total_steps / self.cycles
        self.warmup_steps = self.steps_per_cycle * warmup_ratio
        self.decaying_steps = self.steps_per_cycle - self.warmup_steps
        self.warmup_lr_ratio = warmup_lr_ratio
        self.decay_rate = target_lr_ratio ** (1 / self.decaying_steps)
        self.recycle_steps = np.ceil(np.linspace(0.0, self.total_steps, num=self.cycles, endpoint=False)).astype(int).tolist()
        self.external_factors = dict()
        super(CyclicStepLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.total_steps:
            step_in_cycle = self.last_epoch % self.steps_per_cycle
            if step_in_cycle <= self.warmup_steps:
                lr_factor = self.warmup_lr_ratio + (1 - self.warmup_lr_ratio) * (step_in_cycle / self.warmup_steps)
            else:
                lr_factor = (self.decay_rate ** (step_in_cycle - self.warmup_steps))
        else:
            lr_factor = 1.0
        for factor in self.external_factors.values():
            lr_factor *= factor
        return [base_lr * lr_factor for base_lr in self.base_lrs]
