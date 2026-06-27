import pytorch_lightning as pl
import torch

from utils.plot_utils import (
    apply_correction,
    flatten_model,
    get_full_mask,
    get_perturbation,
)


class GradualProjection(pl.Callback):
    def __init__(
            self,
            trajs: torch.Tensor,
            targ_coeffs: torch.Tensor,
            power: float,
            warmup_ratio: float,
            cooldown_ratio: float,
            ignore_bn: bool = True,
            ignore_bias: bool = True,
            correction_interval: int = 1,
            driver: str = "gelsd",
            name: str = "progress/projection",
        ):
        super().__init__()
        self.trajs = trajs
        self.targ_coeffs = targ_coeffs
        self.power = power
        self.warmup_ratio = warmup_ratio
        self.cooldown_ratio = cooldown_ratio
        self.ignore_bn = ignore_bn
        self.ignore_bias = ignore_bias
        self.correction_interval = correction_interval
        self.driver = driver
        self.name = name

    # The setup should only run after the model has completely finished pruning
    @torch.no_grad()
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        self.mask = get_full_mask(
            pl_module,
            ignore_bn=self.ignore_bn,
            ignore_bias=self.ignore_bias
        )
        device = pl_module.device
        self.mask = self.mask.to(device)
        self.trajs = self.trajs.to(device)
        self.targ_coeffs = self.targ_coeffs.to(device)

    @torch.no_grad()
    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs, batch, batch_idx
        ):

        current_step = trainer.global_step
        if current_step % self.correction_interval == 0:
            progress_factor = current_step / trainer.max_steps
            if progress_factor <= self.warmup_ratio:
                correction_target = 0.0
            elif progress_factor >= (1.0 - self.cooldown_ratio):
                correction_target = 1.0
            else:
                correction_target = 1.0 - 1.0 * (
                    1.0 - (progress_factor - self.warmup_ratio) / (
                        1.0 - self.warmup_ratio - self.cooldown_ratio)
                ) ** self.power

            orig_vec = flatten_model(
                pl_module,
                consider_mask=True,
                ignore_bn=self.ignore_bn,
                ignore_bias=self.ignore_bias,
            )
            pert_vec = get_perturbation(
                self.trajs, orig_vec, self.targ_coeffs, self.mask
            )
            correction = pert_vec * correction_target
            apply_correction(
                pl_module,
                correction,
                ignore_bn=self.ignore_bn,
                ignore_bias=self.ignore_bias
            )

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module):

        # Log L2 norm of coefficient difference
        new_vec = flatten_model(
            pl_module,
            consider_mask=True,
            ignore_bn=self.ignore_bn,
            ignore_bias=self.ignore_bias
        )
        if self.driver == "gels":
            new_coeff = torch.linalg.lstsq(self.trajs.T, new_vec, driver=self.driver).solution
        else:
            # NOTE: torch.linalg.lstsq: `driver` other than `gels` is not supported on CUDA
            new_coeff = torch.linalg.lstsq(self.trajs.T.cpu(), new_vec.cpu(), driver=self.driver).solution.to(new_vec.device)
        pl_module.log(f"{self.name}.l2", torch.norm(new_coeff - self.targ_coeffs, p=2))

        # Log vector correction L2 norm
        # pert_vec = get_perturbation(
        #     self.trajs, orig_vec, self.targ_coeffs, self.mask
        # )
        # pl_module.log(f"{self.name}.l2", torch.norm(pert_vec, p=2))
