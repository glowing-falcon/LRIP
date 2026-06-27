import logging
from pprint import pformat

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

import transforms

from .utils import ImageFolderDict


class ImageFolderImclassV1(pl.LightningDataModule):
    def __init__(self, args, data_config):
        super(ImageFolderImclassV1, self).__init__()
        # self.save_hyperparameters()
        self.args = args
        self.data_config = data_config
        self.rng = np.random.default_rng(self.data_config.dataloader.shuffle_seed)

    def setup(self, stage=None, verbose=True):
        self.train_compose_fn = torchvision.transforms.Compose([
            getattr(transforms, transform_cfg["type"])(transform_cfg["name"], **transform_cfg["params"])
            for transform_cfg in self.data_config.transforms.train
        ])
        self.val_compose_fn = torchvision.transforms.Compose([
            getattr(transforms, transform_cfg["type"])(transform_cfg["name"], **transform_cfg["params"])
            for transform_cfg in self.data_config.transforms.validation
        ])
        train_ds, val_ds, test_ds, self.info = get_dataset(
            **self.data_config.params,
            train_compose_fn=self.train_compose_fn,
            val_compose_fn=self.val_compose_fn,
        )
        if verbose:
            logging.info(f"Dataset info: {pformat(self.info)}")
        if train_ds.classes != val_ds.classes:
            raise ValueError(
                "ImageFolder class lists differ between train and val. "
                f"train={train_ds.classes}, val={val_ds.classes}"
            )
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.dataloader.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,  # Because cutmix wants even bs
            generator=torch.Generator().manual_seed(self.data_config.dataloader.shuffle_seed),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.dataloader.batch_size,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not available")
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.data_config.dataloader.batch_size,
                num_workers=self.args.num_workers,
            )


def get_dataset(
        train_root: str,
        train_compose_fn: torchvision.transforms.Compose,
        val_root: str,
        val_compose_fn: torchvision.transforms.Compose,
        reformat_tts: float = 0.0,
    ) -> tuple[Dataset, Dataset, Dataset | None, dict]:
    if reformat_tts:
        if not (0.0 < reformat_tts < 1.0):
            raise ValueError(f"reformat_tts must be in (0, 1), got {reformat_tts}")

        # Original val split becomes test split.
        test_ds = ImageFolderDict(
            torchvision.datasets.ImageFolder(root=val_root, transform=val_compose_fn)
        )

        # Build two train-root datasets so train/val subsets can use different transforms.
        train_full_train_tf = ImageFolderDict(
            torchvision.datasets.ImageFolder(root=train_root, transform=train_compose_fn)
        )
        train_full_val_tf = ImageFolderDict(
            torchvision.datasets.ImageFolder(root=train_root, transform=val_compose_fn)
        )

        if train_full_train_tf.classes != test_ds.classes:
            raise ValueError(
                "ImageFolder class lists differ between train and test. "
                f"train={train_full_train_tf.classes}, test={test_ds.classes}"
            )

        train_len = len(train_full_train_tf)
        if train_len < 2:
            raise ValueError(f"Train dataset must have at least 2 samples, got {train_len}")

        new_val_len = int(round(train_len * reformat_tts))
        new_val_len = max(1, min(new_val_len, train_len - 1))

        indices = np.random.default_rng(0).permutation(train_len)
        val_indices = indices[:new_val_len].tolist()
        train_indices = indices[new_val_len:].tolist()

        train_ds = Subset(train_full_train_tf, train_indices)
        val_ds = Subset(train_full_val_tf, val_indices)

        # Preserve class metadata expected by setup checks.
        train_ds.classes = train_full_train_tf.classes
        val_ds.classes = train_full_val_tf.classes

        info = {
            "num_classes": len(train_full_train_tf.classes),
            "train_len": len(train_ds),
            "val_len": len(val_ds),
            "test_len": len(test_ds),
        }
        return train_ds, val_ds, test_ds, info
    else:
        train_ds = ImageFolderDict(
            torchvision.datasets.ImageFolder(root=train_root, transform=train_compose_fn)
        )
        val_ds = ImageFolderDict(
            torchvision.datasets.ImageFolder(root=val_root, transform=val_compose_fn)
        )
        info = {
            "num_classes": len(train_ds.classes),
            "train_len": len(train_ds),
            "val_len": len(val_ds),
        }
        return train_ds, val_ds, None, info
