import math
import operator

from pytorch_lightning.callbacks import ModelCheckpoint


class PatientCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        lookout_param,
        comparison,
        *args,
        rel_tol=0.0,
        abs_tol=1e-05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lookout_param = lookout_param
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.comparison = self._resolve_comparison(comparison)

    def _resolve_comparison(self, comparison):
        if callable(comparison):
            return comparison
        if isinstance(comparison, (list, tuple)) and len(comparison) == 2:
            op_name, target = comparison
            op = self._operator_from_name(op_name)
            if op is None:
                raise ValueError(f"Unsupported comparison operator: {op_name}")
            return lambda value: op(value, target)
        raise ValueError("comparison must be callable or [op_name, target]")

    def _is_close(self, value, target):
        try:
            return math.isclose(
                value, target, rel_tol=self.rel_tol, abs_tol=self.abs_tol
            )
        except TypeError:
            return False

    def _operator_from_name(self, name):
        if name is None:
            return None
        mapping = {
            "__ge__": lambda value, target: (
                operator.ge(value, target) or self._is_close(value, target)
            ),
            "__gt__": operator.gt,
            "__le__": lambda value, target: (
                operator.le(value, target) or self._is_close(value, target)
            ),
            "__lt__": operator.lt,
            "__eq__": lambda value, target: self._is_close(value, target),
            "__ne__": lambda value, target: not self._is_close(value, target),
        }
        return mapping.get(name)

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

    def _maybe_update_condition(self, trainer):
        value = self._extract_metric(trainer)
        if value is None:
            return False
        try:
            met = bool(self.comparison(value))
        except TypeError:
            met = bool(self.comparison(value, trainer))
        return met

    def on_validation_end(self, trainer, pl_module):
        if self._maybe_update_condition(trainer):
            super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._maybe_update_condition(trainer):
            super().on_train_epoch_end(trainer, pl_module)
