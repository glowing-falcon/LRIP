
import pytorch_lightning as pl
import torch

from .abc_magnitude import ABCMagnitudePruner


class AutomatedGradualPruningScoreBased(ABCMagnitudePruner):

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
        # prune_frequency = self.prune_frequency / trainer.max_steps

        importance_scores = self.compute_importance_scores(rescore=self.rescore)

        if progress_factor <= self.prune_warmup_ratio:
            prune_target = 0.0
        elif progress_factor >= (1 - self.prune_cooldown_ratio):
            prune_target = self.prune_target
        else:
            if current_step % self.prune_frequency != 0:
                self.skip_pruning(pl_module)
                return
            # rounded_progress = math.ceil(current_step / self.prune_frequency) * prune_frequency
            scorebased_target = 1 - (
                1 - (progress_factor - self.prune_warmup_ratio) / (
                    1 - self.prune_warmup_ratio - self.prune_cooldown_ratio
                )
            ) ** self.prune_power

            flattened_scores = torch.cat([score.view(-1) for score in importance_scores.values()])
            untouchable_score = torch.quantile(flattened_scores, self.prune_target, interpolation="higher")
            min_nonzero = flattened_scores[flattened_scores > 0].min()
            # cutoff_score = scorebased_target * untouchable_score
            cutoff_score = min_nonzero + (untouchable_score - min_nonzero) * scorebased_target
            if self.monitor_prefix:
                pl_module.log(f"{self.monitor_prefix}/cutoff_score", cutoff_score)
            prune_target = (
                cutoff_score >= flattened_scores
            ).sum().item() / len(flattened_scores)

        self.prune(
            importance_scores,
            prune_target,
            pl_module,
            rescore=self.rescore
        )
