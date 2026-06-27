import math

import pytorch_lightning as pl


class LRBasedAdjust(pl.Callback):

    def __init__(
        self,
        name: str = "lr_factor",
        lookout_param: str = "perturbation",
        momentum: float = 0.9,
        lr_factor_cap: float = 10.0,
    ):
        super().__init__()
        self.name = name
        self.lookout_param = lookout_param
        self.momentum = momentum
        self.lr_factor = 1.0
        self.lr_factor_cap = lr_factor_cap

    def _extract_metric(self, trainer):
        metrics = trainer.callback_metrics
        if self.lookout_param not in metrics:
            return None
        value = metrics[self.lookout_param]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        return value

    # This step should come after pruning, but before any backprop
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        metric = self._extract_metric(trainer)
        base_lr = trainer.lr_scheduler_configs[0].scheduler.base_lrs[0]
        multiplier = (self.lr_factor_cap - 1) * math.tanh(metric / base_lr) + 1
        # Performing this, though technically multiplier should not exceed lr_factor_cap
        # due to how tanh works.
        self.lr_factor = min(
            self.momentum * self.lr_factor + (1 - self.momentum) * multiplier,
            self.lr_factor_cap
        )
        pl_module.log(self.name, self.lr_factor)
        lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
        lr_scheduler.external_factors[self.name] = self.lr_factor
