import math
from copy import deepcopy
from typing import List

import pytorch_lightning as pl

from .abc_wanda import ABCWandaPruner


class LearningRateIntegralPruningWanda(ABCWandaPruner):

    schedule: List[float]

    def __init__(
            self,
            prune_power: float,
            prune_cooldown_ratio: float,
            prune_frequency: int = 1,
            rescore: bool = False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.prune_power = prune_power
        self.prune_cooldown_ratio = prune_cooldown_ratio
        self.prune_frequency = prune_frequency
        self.rescore = rescore

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_start(trainer, pl_module)
        scheduler_copy = deepcopy(trainer.lr_scheduler_configs[0].scheduler)
        self.schedule = []
        for _ in range(trainer.max_steps):
            scheduler_copy.step()
            lr = scheduler_copy.get_last_lr()[0]
            self.schedule.append(lr ** self.prune_power)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        current_step = trainer.global_step
        end_step = round((1 - self.prune_cooldown_ratio) * trainer.max_steps)
        if current_step < end_step:
            rounded_current_step = math.ceil(current_step / self.prune_frequency) * self.prune_frequency
            lr_area = sum(self.schedule[:end_step])
            lr_curr_area = sum(self.schedule[:rounded_current_step])
            prune_target = self.prune_target * (lr_curr_area / lr_area)
        else:
            prune_target = self.prune_target

        self.prune(
            self.compute_importance_scores(),
            prune_target,
            pl_module,
            rescore=self.rescore
        )
