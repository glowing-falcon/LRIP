import logging
import math
from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.utils.prune as prune

from utils.plot_utils import multi_dim_norm

from .abc_pruner import ABCPruner


# NOTE: Only linear layers are supported for now since we need to compute per-row importance scores,
# but the framework can be extended to support other layer types as well.
class ABCWandaPruner(ABCPruner, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activation_norms = {}
        self._hooks = []

    # --- Hook lifecycle ---

    def register_calibration_hooks(self):
        for key, (module, attr) in self.target_moduledict.items():
            def make_hook(k):
                def hook(mod, inp, out):
                    x = inp[0]
                    # logging.info(f"Registering activation norms for [{k}]: {x.shape}")
                    self._activation_norms[k] = multi_dim_norm(x, dim=list(range(0, x.dim() - 1))).detach()
                    # self._activation_norms[k] = x.norm(p=2, dim=0).detach()
                return hook
            # logging.info(f"Registering calibration hook for [{key}]")
            self._hooks.append(module.register_forward_hook(make_hook(key)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # --- pl.Callback ---

    def on_train_start(self, trainer, pl_module: pl.LightningModule):
        self.register_calibration_hooks()

    def on_train_end(self, trainer, pl_module: pl.LightningModule):
        self.remove_hooks()

    # --- Wanda importance scoring ---

    def compute_importance_scores(self, rescore=False):
        importance_scores = {}
        for key, (module, attr) in self.target_moduledict.items():
            if rescore:
                for name, param in module.named_parameters():
                    if f"{attr}_orig" == name:
                        weights = getattr(module, f"{attr}_orig")
                        break
                else:
                    raise ValueError(f"Attribute {attr} not found in module {module}")
            else:
                weights = getattr(module, attr)

            if key in self._activation_norms:
                # logging.info(f"Computing importance scores for [{key}] using Wanda method.")
                # Shapes
                # logging.info(f"Weight shape: {weights.shape}, Activation norm shape: {self._activation_norms[key].shape}")
                score = weights.abs() * self._activation_norms[key].unsqueeze(0)  # (C_out, C_in)
            else:
                raise RuntimeError(
                    f"No activation norms found for [{key}]. "
                    "Ensure register_calibration_hooks() was called before the forward pass."
                )

            importance_scores[(module, attr)] = score

        return importance_scores

    # --- Per-output pruning (overrides ABCPruner._prune) ---

    def _prune(self, importance_scores, amount, pl_module: pl.LightningModule, rescore=False):
        for key, (module, name) in self.target_moduledict.items():
            mask = getattr(module, f"{name}_mask")

            if rescore:
                setattr(module, f"{name}_mask", torch.ones_like(mask))
            else:
                if importance_scores[(module, name)].min() < 0:
                    logging.warning(
                        f"Importance scores for [{key}] contains negative values, "
                        "this may lead to unexpected pruning behavior."
                    )
                importance_scores[(module, name)] *= mask
                setattr(module, f"{name}_mask", torch.ones_like(mask))  # in order to retain original weights
            torch.nn.utils.prune.remove(module, name)

            # per-row: prune `amount` fraction of weights per output neuron
            score = importance_scores[(module, name)]
            n_prune = math.ceil(score.shape[1] * amount)
            prune_idx = score.argsort(dim=1)[:, :n_prune]
            new_mask = torch.ones_like(mask)
            new_mask.scatter_(1, prune_idx, 0)

            prune.identity(module, name=name)  # re-register mask
            # manual pruning since global_unstructured doesn't support per-row pruning
            setattr(module, f"{name}_mask", new_mask)
