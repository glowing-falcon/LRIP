import logging
import math
from abc import ABC
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.utils.prune as prune

from utils import get_modules


class ABCPruner(pl.Callback, ABC):

    target_moduledict = None

    def __init__(
            self,
            module_include: List[str],
            module_ignore: List[str],
            prune_target: float,
            monitor_prefix: str = None,
        ):
        super().__init__()
        self.module_include = module_include
        self.module_ignore = module_ignore
        self.prune_target = prune_target
        self.monitor_prefix = monitor_prefix

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        self.target_moduledict = get_modules(
            pl_module.model,
            include=self.module_include,
            ignore=self.module_ignore
        )
        # create temporary masks
        for module, param_name in self.target_moduledict.values():
            if not hasattr(module, f"{param_name}_mask"):
                prune.identity(module, name=param_name)

    def _prune(self, importance_scores, amount, pl_module: pl.LightningModule, rescore=False):
        total_elements = 0
        for key, (module, name) in self.target_moduledict.items():
            total_elements += getattr(module, name).nelement()
            # remove masks first or else it will cause OOM
            mask = getattr(module, f"{name}_mask")
            if rescore:
                setattr(module, f"{name}_mask", torch.ones_like(mask))  # in order to retain original weights
            else:
                # NOTE: this will not work if importance_scores are negative as previously masked weights have a score of 0
                if importance_scores[(module, name)].min() < 0:
                    logging.warning(
                        f"Importance scores for [{key}] contains negative values, "
                        "this may lead to unexpected pruning behavior."
                    )
                importance_scores[(module, name)] *= mask
                setattr(module, f"{name}_mask", torch.ones_like(mask))  # in order to retain original weights

            torch.nn.utils.prune.remove(module, name)
        prune.global_unstructured(
            self.target_moduledict.values(),
            pruning_method=prune.L1Unstructured,
            importance_scores=importance_scores,
            amount=math.ceil(amount * total_elements),
        )

    def prune(self, importance_scores, amount, pl_module: pl.LightningModule, rescore=False):
        if self.monitor_prefix:
            prev_masks = {}
            for key, (module, name) in self.target_moduledict.items():
                mask = getattr(module, f"{name}_mask")
                prev_masks[key] = mask.clone()

        self._prune(importance_scores, amount, pl_module, rescore=rescore)

        if self.monitor_prefix:
            perturbation_normed_squared_sum = 0.0
            for key, (module, name) in self.target_moduledict.items():
                new_mask = getattr(module, f"{name}_mask")
                prev_mask = prev_masks[key]
                weights = getattr(module, f"{name}_orig")
                perturbation_normed_squared_sum += torch.sum(
                    ((weights / weights.norm()) * (prev_mask - new_mask)) ** 2
                ).item()
            perturbation_normed = perturbation_normed_squared_sum / len(self.target_moduledict)
            pl_module.log(f"{self.monitor_prefix}/normed_perturbation", perturbation_normed)

    def skip_pruning(self, pl_module: pl.LightningModule):
        if self.monitor_prefix:
            # pl_module.log(f"{self.monitor_prefix}/perturbation", 0.0)
            pl_module.log(f"{self.monitor_prefix}/normed_perturbation", 0.0)
