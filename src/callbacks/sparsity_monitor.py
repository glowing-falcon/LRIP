from typing import List

import pytorch_lightning as pl
import torch

from utils import get_modules


class SparsityMonitor(pl.Callback):
    def __init__(
            self,
            module_include: List[str],
            module_ignore: List[str],
            name: str = "sparsity",
        ):
        super().__init__()
        self.name = name
        self.module_include = module_include
        self.module_ignore = module_ignore

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        self.target_moduledict = get_modules(
            pl_module.model,
            include=self.module_include,
            ignore=self.module_ignore
        )

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        pruned_elements, total_elements = 0, 0
        for module, param_name in self.target_moduledict.values():
            # This doesnt calculate properly
            # pruned_elements += torch.sum(getattr(module, param_name) == 0).item()
            total_elements += getattr(module, param_name).nelement()
            mask = getattr(module, f"{param_name}_mask", None)
            if mask is not None:
                pruned_elements += torch.sum(mask == 0).item()
        pl_module.log(self.name, pruned_elements / total_elements)
