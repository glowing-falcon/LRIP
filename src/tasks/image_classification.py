import logging
from pprint import pformat

import pytorch_lightning as pl
import torchsummary
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import metrics
import models
import optimizers
import schedulers


class ImageClassification(pl.LightningModule):
    def __init__(self, args, task_config, dataset_info, verbose=True):
        super(ImageClassification, self).__init__()
        # self.save_hyperparameters()
        self.args = args
        self.task_config = task_config

        self.model = getattr(models, self.task_config.model.type)(dataset_info, **self.task_config.model.params)
        if verbose:
            logging.info(f"Model: \n{pformat(torchsummary.summary(self.model, input_size=self.task_config.model_input, verbose=0))}")

        self.loss_fn = getattr(metrics, self.task_config.loss.type)(**self.task_config.loss.params)
        # logging.info(f"Loss: {self.loss_fn}")

        self.metrics_dict = {
            metric_cfg["name"]: getattr(metrics, metric_cfg["type"])(**metric_cfg["params"])
            for metric_cfg in self.task_config.metrics
        }
        # logging.info(f"Metrics: {self.metrics_dict}")

        if self.task_config.cutmix:
            self.cutmix = Mixup(
                **self.task_config.cutmix,
                num_classes=dataset_info["num_classes"],
            )
            self.cutmix_off_timing = getattr(self.task_config, "cutmix_off_timing", 0.75)
            self.train_loss_fn = SoftTargetCrossEntropy()
        else:
            self.cutmix = None
            self.train_loss_fn = self.loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        progress_factor = self.global_step / self.trainer.max_steps
        if (self.cutmix is not None) and (progress_factor < self.task_config.cutmix_off_timing):
            x, y = self.cutmix(x, y)
            pred = self(x)
            loss = self.train_loss_fn(pred, y)
        else:
            pred = self(x)
            loss = self.loss_fn(pred, y)
        metric_results = {f"train/{name}": metric(pred, y) for name, metric in self.metrics_dict.items()}
        metric_results["train/loss"] = loss
        self.log_dict(metric_results, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        pred = self(x)
        loss = self.loss_fn(pred, y)
        metric_results = {f"val/{name}": metric(pred, y) for name, metric in self.metrics_dict.items()}
        metric_results["val/loss"] = loss
        self.log_dict(metric_results, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        pred = self(x)
        loss = self.loss_fn(pred, y)
        metric_results = {f"test/{name}": metric(pred, y) for name, metric in self.metrics_dict.items()}
        metric_results["test/loss"] = loss
        self.log_dict(metric_results, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(optimizers, self.task_config.optimizer.type)(
            params=self.model.parameters(), **self.task_config.optimizer.params
        )
        if self.task_config.scheduler is None:
            return {"optimizer": optimizer}
        else:
            scheduler = getattr(schedulers, self.task_config.scheduler.type)(
                optimizer=optimizer, **self.task_config.scheduler.params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.task_config.scheduler.config_params,
                },
            }
