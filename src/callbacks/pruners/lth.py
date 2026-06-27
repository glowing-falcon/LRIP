import logging
from copy import deepcopy
from pprint import pformat

import pytorch_lightning as pl
import torch

from .abc_magnitude import ABCMagnitudePruner


class LotteryTicketHypothesis(ABCMagnitudePruner):

    def __init__(
            self,
            rescore: bool = False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.rescore = rescore
        self.original_state_dict: dict[str, torch.Tensor] = dict()
        self.reset_step_targets: dict[int, float] = dict()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.original_state_dict = deepcopy(pl_module.state_dict())
        scheduler_copy = deepcopy(trainer.lr_scheduler_configs[0].scheduler)
        lr_schedule = []
        for _ in range(trainer.max_steps):
            scheduler_copy.step()
            lr = scheduler_copy.get_last_lr()[0]
            lr_schedule.append(lr)
        reset_schedule = [False] + [(
                lr_schedule[i] < lr_schedule[i + 1]
            ) and (
                lr_schedule[i] < lr_schedule[i - 1]
            ) for i in range(1, len(lr_schedule) - 1)
        ] + [False]
        reset_steps = [i for i, x in enumerate(reset_schedule) if x]
        self.reset_step_targets = {
            step: 1 - ((1 - self.prune_target) ** ((reset_steps.index(step) + 1) / len(reset_steps))) for step in reset_steps
        }
        logging.info(f"Total steps: {scheduler_copy.total_steps}")
        logging.info("Lottery Ticket Hypothesis reset steps:")
        logging.info(pformat(self.reset_step_targets))

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx):
        current_step = trainer.global_step + 1  # +1 because it is a 'before' step

        if current_step in self.reset_step_targets.keys():
            logging.info(f"Resetting weights at step {current_step} for Lottery Ticket Hypothesis")
            importance_scores = self.compute_importance_scores()

            # reset weights
            for name, param in self.original_state_dict.items():
                if name in pl_module.state_dict():
                    pl_module.state_dict()[name].data.copy_(param.data)

            self.prune(
                importance_scores,
                self.reset_step_targets[current_step],
                pl_module,
                rescore=self.rescore
            )
