import logging
import math
from copy import deepcopy
from typing import List

import pytorch_lightning as pl

from .abc_magnitude import ABCMagnitudePruner


class CyclicLearningRateIntegralPruning(ABCMagnitudePruner):

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
        self.reset_steps: list[int] = []

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        scheduler_copy = deepcopy(trainer.lr_scheduler_configs[0].scheduler)
        self.schedule = []
        for _ in range(trainer.max_steps):
            scheduler_copy.step()
            lr = scheduler_copy.get_last_lr()[0]
            self.schedule.append(lr ** self.prune_power)
        reset_schedule = [False] + [(
                self.schedule[i] < self.schedule[i + 1]
            ) and (
                self.schedule[i] < self.schedule[i - 1]
            ) for i in range(1, len(self.schedule) - 1)
        ] + [False]
        self.reset_steps = [i for i, x in enumerate(reset_schedule) if x]
        logging.info(f"Total steps: {scheduler_copy.total_steps}")
        logging.info("Cyclic Learning Rate Integral Pruning reset steps:")
        logging.info(self.reset_steps)

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        current_step = trainer.global_step + 1  # +1 because it is a 'before' step
        prev_reset_step = max(filter(lambda x: x <= current_step, self.reset_steps), default=0)
        next_reset_step = min(filter(lambda x: x > current_step, self.reset_steps), default=trainer.max_steps)
        end_step = round((1 - self.prune_cooldown_ratio) * (next_reset_step - prev_reset_step)) + prev_reset_step
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
