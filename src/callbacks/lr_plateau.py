import logging
from typing import Literal

import pytorch_lightning as pl


class LRPlateau(pl.Callback):
    def __init__(
        self,
        name: str = "progress/lr_plateau",
        lookout_param: str = "val/loss",
        mode: Literal["min", "max"] = "min",
        factor: float = 0.5,
        warmup_ratio: float = 0.0,
        patience: int = 5,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_factor: float = 1e-3,
        logging: bool = True,
    ):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")
        if not (0.0 < factor < 1.0):
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if not (0.0 <= warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, got {cooldown}")
        if not (0.0 < min_factor <= 1.0):
            raise ValueError(f"min_factor must be in (0, 1], got {min_factor}")

        self.name = name
        self.lookout_param = lookout_param
        self.mode = mode
        self.factor = factor
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_factor = min_factor
        self.logging = logging

        self.lr_factor = 1.0
        self.best = None
        self.bad_epochs = 0
        self.cooldown_counter = 0

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

    def _is_better(self, value):
        if self.best is None:
            return True
        if self.threshold_mode == "rel":
            if self.mode == "min":
                return value < self.best * (1.0 - self.threshold)
            return value > self.best * (1.0 + self.threshold)
        if self.mode == "min":
            return value < self.best - self.threshold
        return value > self.best + self.threshold

    def _update_lr_factor(self):
        self.lr_factor = max(self.lr_factor * self.factor, self.min_factor)

    def _in_warmup(self, trainer: pl.Trainer):
        if self.warmup_ratio <= 0.0:
            return False
        if trainer.max_steps is None or trainer.max_steps <= 0:
            logging.warning("LRPlateau: max_steps is not set or non-positive. Skipping warmup.")
            return False
        return trainer.global_step < int(self.warmup_ratio * trainer.max_steps)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        value = self._extract_metric(trainer)
        if value is None:
            logging.warning(f"LRPlateau: Metric '{self.lookout_param}' not found. Skipping LR adjustment.")
            return

        if not self._in_warmup(trainer):
            if self._is_better(value):
                self.best = value
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            elif self.bad_epochs > self.patience:
                self._update_lr_factor()
                self.cooldown_counter = self.cooldown

        if len(trainer.lr_scheduler_configs) == 0:
            raise ValueError("LRPlateau requires at least one scheduler with `external_factors`.")
        lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
        if not hasattr(lr_scheduler, "external_factors"):
            raise ValueError(
                "LRPlateau requires scheduler.external_factors. "
                "Use a compatible scheduler such as ListenerLR/CyclicStepLR."
            )
        lr_scheduler.external_factors[self.name] = self.lr_factor
        if self.logging:
            pl_module.log(self.name, self.lr_factor, on_step=False, on_epoch=True)
