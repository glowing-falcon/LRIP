import math
from copy import deepcopy
from typing import List

import pytorch_lightning as pl
import torch

from .abc_magnitude import ABCMagnitudePruner


class LearningRateIntegralPruningScoreBased(ABCMagnitudePruner):

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
        scheduler_copy = deepcopy(trainer.lr_scheduler_configs[0].scheduler)
        self.schedule = []
        for _ in range(trainer.max_steps):
            scheduler_copy.step()
            lr = scheduler_copy.get_last_lr()[0]
            self.schedule.append(lr ** self.prune_power)

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        current_step = trainer.global_step + 1  # +1 because it is a 'before' step
        importance_scores = self.compute_importance_scores(rescore=self.rescore)
        end_step = round((1 - self.prune_cooldown_ratio) * trainer.max_steps)
        if current_step < end_step:
            rounded_current_step = math.ceil(current_step / self.prune_frequency) * self.prune_frequency
            lr_area = sum(self.schedule[:end_step])
            lr_curr_area = sum(self.schedule[:rounded_current_step])
            scorebased_target = lr_curr_area / lr_area

            flattened_scores = torch.cat([score.view(-1) for score in importance_scores.values()])
            untouchable_score = torch.quantile(flattened_scores, self.prune_target, interpolation="higher")
            min_nonzero = flattened_scores[flattened_scores > 0].min()
            # cutoff_score = scorebased_target * untouchable_score
            cutoff_score = min_nonzero + (untouchable_score - min_nonzero) * scorebased_target
            if self.monitor_prefix:
                pl_module.log(f"{self.monitor_prefix}/cutoff_score", cutoff_score)
            prune_target = (
                cutoff_score >= flattened_scores
            ).sum().item() / len(flattened_scores)

        else:
            prune_target = self.prune_target

        self.prune(
            importance_scores,
            prune_target,
            pl_module,
            rescore=self.rescore
        )
