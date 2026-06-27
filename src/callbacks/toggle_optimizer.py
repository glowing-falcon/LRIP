import pytorch_lightning as pl


class ToggleOptimizer(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for optimizer in trainer.optimizers:
            optimizer.train()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for optimizer in trainer.optimizers:
            optimizer.eval()
