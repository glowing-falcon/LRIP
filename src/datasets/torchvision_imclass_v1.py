import copy
import logging
from pprint import pformat

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

import transforms

from .utils import ConcatWithClasses, ImageFolderDict, MNISTInMemory


def _clone_with_transform(source_dataset: Dataset, transform: torchvision.transforms.Compose) -> Dataset:
    """
    Best-effort clone that swaps transform while preserving wrappers.
    """
    if hasattr(source_dataset, "base_ds"):
        cloned = copy.copy(source_dataset)
        base_ds = source_dataset.base_ds

        if hasattr(base_ds, "transform"):
            cloned.base_ds = copy.copy(base_ds)
            cloned.base_ds.transform = transform
            return cloned

        if hasattr(base_ds, "datasets"):
            cloned_base = copy.copy(base_ds)
            cloned_base.datasets = []
            for ds in base_ds.datasets:
                ds_clone = copy.copy(ds)
                if hasattr(ds_clone, "transform"):
                    ds_clone.transform = transform
                cloned_base.datasets.append(ds_clone)
            if hasattr(base_ds, "classes"):
                cloned_base.classes = base_ds.classes
            cloned.base_ds = cloned_base
            return cloned

    if hasattr(source_dataset, "transform"):
        cloned = copy.copy(source_dataset)
        cloned.transform = transform
        return cloned

    return source_dataset


class TorchvisionImclassV1(pl.LightningDataModule):
    def __init__(self, args, data_config):
        super(TorchvisionImclassV1, self).__init__()
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
                "Class lists differ between train and val. "
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
            persistent_workers=self.args.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.dataloader.batch_size,
            num_workers=self.args.num_workers,
            persistent_workers=self.args.num_workers > 0,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset is not available")
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.data_config.dataloader.batch_size,
                num_workers=self.args.num_workers,
                persistent_workers=self.args.num_workers > 0,
            )


def get_dataset(
        name: str,
        train_compose_fn: torchvision.transforms.Compose,
        val_compose_fn: torchvision.transforms.Compose,
        reformat_tts: float = 0.0,
        **kwargs,
    ) -> tuple[Dataset, Dataset, Dataset | None, dict]:
    match name:
        case "CIFAR10":
            train_ds = ImageFolderDict(torchvision.datasets.CIFAR10(
                train=True,
                download=False,
                transform=train_compose_fn,
                **kwargs,
            ))
            val_ds = ImageFolderDict(torchvision.datasets.CIFAR10(
                train=False,
                download=False,
                transform=val_compose_fn,
                **kwargs,
            ))
        case "CIFAR100":
            train_ds = ImageFolderDict(torchvision.datasets.CIFAR100(
                train=True,
                download=False,
                transform=train_compose_fn,
                **kwargs,
            ))
            val_ds = ImageFolderDict(torchvision.datasets.CIFAR100(
                train=False,
                download=False,
                transform=val_compose_fn,
                **kwargs,
            ))
        case "OxfordIIITPet":
            train_ds = ImageFolderDict(torchvision.datasets.OxfordIIITPet(
                split="trainval",
                download=False,
                transform=train_compose_fn,
                **kwargs,
            ))
            val_ds = ImageFolderDict(torchvision.datasets.OxfordIIITPet(
                split="test",
                download=False,
                transform=val_compose_fn,
                **kwargs,
            ))
        case "Flowers102":
            # Uses train and val splits for training, test split for validation
            train_ds = ImageFolderDict(ConcatWithClasses([torchvision.datasets.Flowers102(
                split="train",
                download=False,
                transform=train_compose_fn,
                **kwargs,
            ), torchvision.datasets.Flowers102(
                split="val",
                download=False,
                transform=train_compose_fn,
                **kwargs,
            )]))
            val_ds = ImageFolderDict(torchvision.datasets.Flowers102(
                split="test",
                download=False,
                transform=val_compose_fn,
                **kwargs,
            ))
        case "StanfordCars":
            train_ds = ImageFolderDict(torchvision.datasets.StanfordCars(
                split="train",
                download=False,
                transform=train_compose_fn,
                **kwargs,
            ))
            val_ds = ImageFolderDict(torchvision.datasets.StanfordCars(
                split="test",
                download=False,
                transform=val_compose_fn,
                **kwargs,
            ))
        case "Food101":
            train_ds = ImageFolderDict(torchvision.datasets.Food101(
                split="train",
                download=False,
                transform=train_compose_fn,
                **kwargs,
            ))
            val_ds = ImageFolderDict(torchvision.datasets.Food101(
                split="test",
                download=False,
                transform=val_compose_fn,
                **kwargs,
            ))
        # case "SUN397":
        #     train_ds = ImageFolderDict(torchvision.datasets.SUN397(
        #         split="train",
        #         download=False,
        #         transform=train_compose_fn,
        #         **kwargs,
        #     ))
        #     val_ds = ImageFolderDict(torchvision.datasets.SUN397(
        #         split="test",
        #         download=False,
        #         transform=val_compose_fn,
        #         **kwargs,
        #     ))
        case "MNIST":
            assert len(train_compose_fn.transforms) == 0, "Transform is not supported for our MNIST."
            assert len(val_compose_fn.transforms) == 0, "Transform is not supported for our MNIST."
            train_ds = ImageFolderDict(MNISTInMemory(torchvision.datasets.MNIST(
                train=True,
                download=True,
                transform=None,
                **kwargs,
            )))
            val_ds = ImageFolderDict(MNISTInMemory(torchvision.datasets.MNIST(
                train=False,
                download=True,
                transform=None,
                **kwargs,
            )))
        case "FashionMNIST":
            assert len(train_compose_fn.transforms) == 0, "Transform is not supported for our MNIST."
            assert len(val_compose_fn.transforms) == 0, "Transform is not supported for our MNIST."
            train_ds = ImageFolderDict(MNISTInMemory(torchvision.datasets.FashionMNIST(
                train=True,
                download=True,
                transform=None,
                **kwargs,
            )))
            val_ds = ImageFolderDict(MNISTInMemory(torchvision.datasets.FashionMNIST(
                train=False,
                download=True,
                transform=None,
                **kwargs,
            )))
        case _:
            raise ValueError(f"Unsupported dataset: {name}")

    if reformat_tts:
        if not (0.0 < reformat_tts < 1.0):
            raise ValueError(f"reformat_tts must be in (0, 1), got {reformat_tts}")

        # Original val split becomes test split.
        test_ds = val_ds

        # Prefer cloning train dataset and swapping transform to avoid reinitializing datasets.
        # For datasets without transform support (e.g., MNISTInMemory), this returns source as-is.
        train_full_val_tf = _clone_with_transform(train_ds, val_compose_fn)

        if train_ds.classes != test_ds.classes:
            raise ValueError(
                "Class lists differ between train and test. "
                f"train={train_ds.classes}, test={test_ds.classes}"
            )

        train_len = len(train_ds)
        if train_len < 2:
            raise ValueError(f"Train dataset must have at least 2 samples, got {train_len}")

        new_val_len = int(round(train_len * reformat_tts))
        new_val_len = max(1, min(new_val_len, train_len - 1))

        indices = np.random.default_rng(0).permutation(train_len)
        val_indices = indices[:new_val_len].tolist()
        train_indices = indices[new_val_len:].tolist()

        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(train_full_val_tf, val_indices)

        # Preserve class metadata expected by setup checks.
        train_ds.classes = train_full_val_tf.classes
        val_ds.classes = train_full_val_tf.classes

        info = {
            "num_classes": len(train_full_val_tf.classes),
            "train_len": len(train_ds),
            "val_len": len(val_ds),
            "test_len": len(test_ds),
        }
        return train_ds, val_ds, test_ds, info
    else:
        info = {
            "num_classes": len(train_ds.classes),
            "train_len": len(train_ds),
            "val_len": len(val_ds),
        }
        return train_ds, val_ds, None, info
