import math

import pytorch_lightning as pl

from .abc_magnitude import ABCMagnitudePruner


class AutomatedGradualPruning(ABCMagnitudePruner):

    def __init__(
            self,
            prune_power: float,
            prune_warmup_ratio: float,
            prune_cooldown_ratio: float,
            prune_frequency: int = 1,
            rescore: bool = False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.prune_power = prune_power
        self.prune_warmup_ratio = prune_warmup_ratio
        self.prune_cooldown_ratio = prune_cooldown_ratio
        self.prune_frequency = prune_frequency
        self.rescore = rescore

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        current_step = trainer.global_step + 1  # +1 because it is a 'before' step
        progress_factor = current_step / trainer.max_steps
        prune_frequency = self.prune_frequency / trainer.max_steps
        if progress_factor <= self.prune_warmup_ratio:
            prune_target = 0.0
        elif progress_factor >= (1 - self.prune_cooldown_ratio):
            prune_target = self.prune_target
        else:
            rounded_progress = math.ceil(current_step / self.prune_frequency) * prune_frequency
            prune_target = self.prune_target + (
                0.0 - self.prune_target
            ) * (
                1 - (rounded_progress - self.prune_warmup_ratio) / (
                    1 - self.prune_warmup_ratio - self.prune_cooldown_ratio
                )
            ) ** self.prune_power

        self.prune(
            self.compute_importance_scores(),
            prune_target,
            pl_module,
            rescore=self.rescore
        )
