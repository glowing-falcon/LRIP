import logging
from typing import Literal

import pytorch_lightning as pl


class LRMomentum(pl.Callback):
    def __init__(
        self,
        name: str = "progress/lr_momentum",
        lookout_param: str = "val/loss",
        mode: Literal["min", "max"] = "min",
        factor: float = 0.9,
        momentum: float = 0.9,
        warmup: int = 3
    ):
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.name = name
        self.lookout_param = lookout_param
        self.mode = mode
        self.factor = factor
        self.momentum = momentum
        self.warmup = warmup

        self.lr_factor = 1.0
        self.running_value = None
        self.warmup_counter = 0

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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        value = self._extract_metric(trainer)
        if value is None:
            logging.warning(f"Metric not found: {self.lookout_param}")
            return

        # Update running value
        if self.running_value is None:
            self.running_value = value
        else:
            self.running_value = self.momentum * self.running_value + (1 - self.momentum) * value

        if self.warmup_counter < self.warmup:
            self.warmup_counter += 1
        else:
            if self.mode == "min":
                if value > self.running_value:
                    self.lr_factor *= self.factor
            elif self.mode == "max":
                if value < self.running_value:
                    self.lr_factor *= self.factor
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        if len(trainer.lr_scheduler_configs) == 0:
            raise ValueError("LRMomentum requires at least one scheduler with `external_factors`.")
        lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
        if not hasattr(lr_scheduler, "external_factors"):
            raise ValueError(
                "LRMomentum requires scheduler.external_factors. "
                "Use a compatible scheduler such as ListenerLR/CyclicStepLR."
            )
        lr_scheduler.external_factors[self.name] = self.lr_factor
        pl_module.log(self.name, self.lr_factor)
        pl_module.log(f"{self.name}.running_value", self.running_value)
