import math

import pytorch_lightning as pl

from .abc_magnitude import ABCMagnitudePruner


class OneShotMagnitudePruning(ABCMagnitudePruner):

    def __init__(
            self,
            prune_timing: float = 0.0,
            **kwargs,
        ):
        super().__init__(**kwargs)
        if not 0.0 <= prune_timing <= 1.0:
            raise ValueError(f"prune_timing must be in [0.0, 1.0], got {prune_timing}")
        self.prune_timing = prune_timing
        self._pruned = False
        self._prune_step = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        max_steps = max(1, trainer.max_steps)
        self._prune_step = max(1, math.ceil(self.prune_timing * max_steps))

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        if self._pruned:
            self.skip_pruning(pl_module)
            return

        current_step = trainer.global_step + 1  # +1 because it is a 'before' step
        if current_step < self._prune_step:
            self.skip_pruning(pl_module)
            return

        self.prune(
            self.compute_importance_scores(),
            self.prune_target,
            pl_module,
        )
        self._pruned = True
