#!/usr/bin/env python3

import argparse
import datetime
import logging
from pathlib import Path
from pprint import pformat

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import callbacks
import datasets
import tasks
from utils import NamedDict, setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level")
    parser.add_argument("--config", type=Path, default=Path("exp/sample.yaml"), help="Path to config file")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
        help="directory to store run outputs")
    parser.add_argument("--run-name", type=str, default=datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S"),
        help="name of the run")
    return parser.parse_args()


def main():
    args = NamedDict(vars(parse_args()))
    (args.runs_dir / args.run_name).mkdir(parents=True, exist_ok=True)
    setup_logging(out_file=args.runs_dir / args.run_name / "train.log", level=getattr(logging, args.loglevel.upper()))
    args.to_yaml(args.runs_dir / args.run_name / "args.yaml")
    config = NamedDict.from_yaml(args.config)
    config.to_yaml(args.runs_dir / args.run_name / "config.yaml")
    logging.info(f"Configuration: {pformat(config)}")

    dataset = getattr(datasets, config.dataset.type)(args, config.dataset)
    dataset.setup()

    task = getattr(tasks, config.task.type)(args, config.task, dataset.info)

    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir=args.runs_dir,
            name=args.run_name,
            version="",
        ),
        pl.loggers.CSVLogger(
            save_dir=args.runs_dir,
            name=args.run_name,
            version="",
        )
    ]

    trainer_callbacks = [
        getattr(callbacks, cb["type"])(cb["name"], **cb["params"]) if "name" in cb else
        getattr(callbacks, cb["type"])(**cb["params"])
        for cb in config.callbacks if cb is not None
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        # log_every_n_steps=1,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=trainer_callbacks,
        **config.trainer,
    )
    trainer.fit(task, datamodule=dataset)

    checkpoint_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
    has_test_dataloader = False
    test_dataloader = getattr(dataset, "test_dataloader", None)
    if callable(test_dataloader):
        try:
            has_test_dataloader = test_dataloader() is not None
        except Exception as exc:
            logging.info(f"Skipping test: failed to build test dataloader ({exc}).")

    if checkpoint_callbacks and has_test_dataloader:
        best_ckpt_path = None
        for cb in checkpoint_callbacks:
            if cb.best_model_path:
                best_ckpt_path = cb.best_model_path
                break

        if not best_ckpt_path or not Path(best_ckpt_path).exists():
            logging.warning(
                "Skipping test because best checkpoint path does not exist: %s",
                best_ckpt_path,
            )
        else:
            logging.info(f"Rnuning test with best checkpoint: {best_ckpt_path}")
            trainer.test(model=task, datamodule=dataset, ckpt_path=best_ckpt_path, weights_only=False)


if __name__ == "__main__":
    main()
