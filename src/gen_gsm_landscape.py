#!/usr/bin/env python3

import dataclasses
import glob
import logging
import math
import multiprocessing as mp
import queue
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from pprint import pformat

import hydra
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

import datasets
import tasks
from callbacks import GradualProjection
from utils import NamedDict, setup_logging_v2
from utils.plot_utils import (
    filter_named_parameters,
    flatten_model,
    get_filter_norm,
    join_mask_and_weights,
    mask_as_state_dict,
    prune_state_dict,
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def extract_config_dir(argv: list[str]) -> Path | None:
    for i, arg in enumerate(argv):
        if arg == "--config-dir" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config-dir="):
            return Path(arg.split("=", 1)[1])
    return None


@torch.no_grad()
def sanity_check(
    run_dirs: list[Path]
) -> pl.LightningModule:

    # NOTE: I might want to phase out NamedDict
    args = NamedDict(num_workers=0)
    keep_same = dict()
    for run_dir in run_dirs:
        config = NamedDict.from_yaml(Path(run_dir) / "config.yaml")

        if keep_same.get("dataset", config.dataset) == config.dataset:
            keep_same["dataset"] = config.dataset
        else:
            raise ValueError("Incompatible datasets found in run directories")

        if keep_same.get("task.type", config.task.type) == config.task.type:
            keep_same["task.type"] = config.task.type
        else:
            raise ValueError("Incompatible tasks found in run directories")

        if keep_same.get("task.model", config.task.model) == config.task.model:
            keep_same["task.model"] = config.task.model
        else:
            raise ValueError("Incompatible task models found in run directories")

    logging.info("Using configuration: %s", pformat(keep_same))
    dataset = getattr(datasets, config.dataset.type)(args, config.dataset)
    dataset.setup()
    task = getattr(tasks, config.task.type)(args, config.task, dataset.info)

    return task


@torch.no_grad()
def step01_fit_pca(
    task: pl.LightningModule,
    run_dirs: list[Path],
    every_n: int,
    ignore_bn: bool,
    ignore_bias: bool,
) -> tuple[PCA, dict[str, np.ndarray]]:
    # NOTE: task MUST be newly generated at this point
    flat_weights_dict = dict()
    step_zero_weights = flatten_model(
        task.model,
        consider_mask=True,
        ignore_bn=ignore_bn,
        ignore_bias=ignore_bias,
    )
    # Extract weights from checkpoints
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        ckpt_dirs = sorted(
            (run_dir / "checkpoints").glob("*.ckpt"),
            key=lambda p: int(re.search(r"step=(\d+)\.ckpt", p.name).group(1))
        )[::every_n]
        flat_weights_dict[run_dir.name] = [step_zero_weights]
        for ckpt_dir in tqdm.tqdm(ckpt_dirs, desc=f"Processing checkpoints for {run_dir.name}"):
            ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
            state_dict = prune_state_dict(ckpt["state_dict"])
            task.load_state_dict(state_dict)
            flat_weights = flatten_model(
                task.model,
                # NOTE: I don't think consider_mask matters here since we already pruned the state dict
                consider_mask=True,
                ignore_bn=ignore_bn,
                ignore_bias=ignore_bias,
            )
            flat_weights_dict[run_dir.name].append(flat_weights)
        flat_weights_dict[run_dir.name] = torch.stack(flat_weights_dict[run_dir.name])
        logging.info(
            "Extracted all ckpt weights for %s shaped: %s",
            run_dir.name, list(flat_weights_dict[run_dir.name].shape),
        )
    pca_weights_dict = dict()
    for run_name, flat_weights in flat_weights_dict.items():
        weights = flat_weights.detach().cpu().numpy()
        # Put origin at final point
        pca_weights_dict[run_name] = weights - weights[-1]
    pca = PCA(n_components=2)
    # pca.fit(np.concatenate(list(pca_weights_dict.values()), axis=0))
    pca.fit(np.concatenate(list(w / np.clip(np.linalg.norm(w, axis=1, keepdims=True), 1e-8, None) for w in pca_weights_dict.values()), axis=0))
    # pca.fit(np.concatenate(list(
    #     flat_weights[:-1] - flat_weights[1:] for flat_weights in pca_weights_dict.values()
    # ), axis=0))
    logging.info("PCA explained variance ratio: %s", pca.explained_variance_ratio_)
    logging.info(
        "Cosine similarity between the 2 PCA components: %f",
        np.dot(pca.components_[0], pca.components_[1]) / (np.linalg.norm(pca.components_[0]) * np.linalg.norm(pca.components_[1])),
    )
    return pca, pca_weights_dict


@torch.no_grad()
def step02_build_vector_map(
    task: pl.LightningModule,
    pca: PCA,
    pca_weights_dict: dict[str, np.ndarray],
    end_point_ckpts: dict[str, Path],
    ignore_bn: bool,
    ignore_bias: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Need to normalize across all final model points
    filter_norm_dict = dict()
    for _run_dir, ckpt_dir in end_point_ckpts.items():
        ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
        state_dict = prune_state_dict(ckpt["state_dict"])
        task.load_state_dict(state_dict)
        for name, param in filter_named_parameters(task.model, ignore_bn=ignore_bn, ignore_bias=ignore_bias):
            if name not in filter_norm_dict:
                filter_norm_dict[name] = []
            try:
                norm = get_filter_norm(param)
            except ValueError as e:
                logging.critical("Error computing filter norm for %s with shape %s", name, list(param.shape))
                raise e
            norm = norm.expand(2, *param.shape)
            filter_norm_dict[name].append(norm)
    for name, norms in filter_norm_dict.items():
        filter_norm_dict[name] = torch.stack(norms, dim=0).norm(dim=0)

    # Build vector map
    vector_map = []
    weight_index = 0
    pca_n_components = pca.components_.shape[0]
    for name, param in filter_named_parameters(task.model, ignore_bn=ignore_bn, ignore_bias=ignore_bias):
        vector = pca.components_[:, weight_index:weight_index + param.numel()]
        vector = vector.reshape(pca_n_components, *param.shape)
        vector = torch.tensor(vector, device="cpu")
        # NOTE: there is probably a better way, I just got lazy
        # These 2 lines would ideally be done in 1 clean line
        vector[0] = vector[0] / get_filter_norm(vector[0])
        vector[1] = vector[1] / get_filter_norm(vector[1])
        vector_map.append(
            torch.flatten(
                torch.tensor(vector.reshape(pca_n_components, *param.shape), device="cpu")
                    * filter_norm_dict[name],
                start_dim=1,
            )
        )
        weight_index += param.numel()
    vector_map = torch.cat(vector_map, dim=1)
    logging.info("Built vector map with shape: %s", list(vector_map.shape))

    # Project onto vector map
    coeff_dict = {}
    for run_name, matrices in pca_weights_dict.items():
        coeff_dict[run_name] = []
        for v in matrices:
            coeff_dict[run_name].append(
                torch.linalg.lstsq(vector_map.T, torch.tensor(v, device="cpu"), driver="gelsd").solution
            )
        coeff_dict[run_name] = torch.stack(coeff_dict[run_name])
        # coeff_dict[run_name] = torch.tensor(matrices, device="cpu") @ vector_map.T
        logging.info("Computed coefficients for %s with shape: %s", run_name, list(coeff_dict[run_name].shape))
    return vector_map, coeff_dict


# def await_gpu(job):
#     global available_gpus
#     while not available_gpus:
#         time.sleep(0.1)
#     device = available_gpus.pop()
#     job["device"] = torch.device(device)
#     try:
#         out = step03_fit(**job)
#     except Exception as e:
#         logging.critical("Error occurred while fitting model for job %s", job["log_name"])
#         logging.critical("Exception: %s", str(e))
#         logging.critical(traceback.format_exc())
#         raise e
#     available_gpus.append(device)
#     return out


# def step03_fit(
#     config: NamedDict,
#     ckpt_dir: Path,
#     vector_map: torch.Tensor,
#     target_coeffs: torch.Tensor,
#     device: torch.device,
#     num_workers: int,
#     ignore_bn: bool,
#     ignore_bias: bool,
#     outputs_dir: Path,
#     log_name: str,
# ) -> Path:

#     donefile = outputs_dir / log_name / "fit.done"
#     if donefile.exists():
#         return donefile

#     args = NamedDict(num_workers=num_workers)
#     dataset = getattr(datasets, config.dataset.type)(args, config.dataset)
#     dataset.setup()
#     task = getattr(tasks, config.task.type)(args, config.task, dataset.info)
#     ckpt = torch.load(ckpt_dir, map_location=device, weights_only=False)
#     state_dict = join_mask_and_weights(ckpt["state_dict"], task.state_dict())
#     mask_as_state_dict(state_dict, task)
#     task.load_state_dict(state_dict)

#     projection_cb = GradualProjection(
#         vector_map,
#         target_coeffs,
#         power=3.0,
#         warmup_ratio=0.1,
#         cooldown_ratio=0.1,
#         ignore_bn=ignore_bn,
#         ignore_bias=ignore_bias,
#     )
#     logger = pl.loggers.CSVLogger(
#         save_dir=outputs_dir,
#         name=log_name,
#         version="",
#     )
#     config.trainer.log_every_n_steps = config.trainer.val_check_interval
#     trainer = pl.Trainer(
#         accelerator="gpu" if device.type == "cuda" else "cpu",
#         devices=1,
#         logger=[logger],
#         callbacks=[projection_cb],
#         enable_model_summary=False,
#         enable_checkpointing=False,
#         **config.trainer,
#     )
#     trainer.fit(task, datamodule=dataset)

#     donefile.touch()
#     return donefile


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg: DictConfig):
    config_dir = extract_config_dir(sys.argv)
    if config_dir is None:
        raise ValueError(
            "Missing --config-dir argument; cannot determine where to write the log file."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = config_dir / "logs" / f"{timestamp}.log"
    setup_logging_v2(log_path)

    logging.info("LOG FILE: \n%s", log_path)
    logging.info("COMMAND: \n%s", " ".join(sys.argv))
    logging.info("CONFIGURATION: \n%s", OmegaConf.to_yaml(cfg))

    run_dirs = [
        run_dir for run_dir_pattern in cfg.run_dirs
        for run_dir in glob.glob(run_dir_pattern)
    ]
    logging.info("RUN DIRECTORIES: \n%s", "\n".join(run_dirs))

    # For later use
    task = sanity_check(run_dirs)
    end_point_ckpts = {
        run_dir: sorted((Path(run_dir) / "checkpoints").glob("*.ckpt"))[-1]
        for run_dir in run_dirs
    }
    outputs_dir = config_dir / "outputs"

    # ===== STEP 01 =====
    # Fit PCA
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if (outputs_dir / "pca.npz").exists() and (outputs_dir / "pca_weights.npz").exists():
        logging.info("Loading existing PCA weights from %s", outputs_dir / "pca.npz")
        pca_data = np.load(outputs_dir / "pca.npz")
        pca = PCA(n_components=2)
        pca.components_ = pca_data["components"]
        pca.explained_variance_ratio_ = pca_data["explained_variance_ratio"]
        pca_weights_dict = dict(np.load(outputs_dir / "pca_weights.npz"))
    else:
        pca, pca_weights_dict = step01_fit_pca(
            task, run_dirs, every_n=cfg.every_n, ignore_bn=cfg.ignore_bn, ignore_bias=cfg.ignore_bias
        )
        np.savez(
            outputs_dir / "pca.npz",
            components=pca.components_,
            explained_variance_ratio=pca.explained_variance_ratio_
        )
        logging.info("PCA results saved to %s", outputs_dir / "pca.npz")
        np.savez_compressed(
            outputs_dir / "pca_weights.npz",
            **pca_weights_dict
        )
        logging.info("PCA weights saved to %s", outputs_dir / "pca_weights.npz")

    # ===== STEP 02 =====
    # Build vector map
    if (outputs_dir / "vector_map_and_coeffs.pt").exists():
        logging.info("Loading existing vector map and coefficients from %s", outputs_dir / "vector_map_and_coeffs.pt")
        vector_map, coeff_dict = torch.load(outputs_dir / "vector_map_and_coeffs.pt")
    else:
        vector_map, coeff_dict = step02_build_vector_map(
            task, pca, pca_weights_dict, end_point_ckpts, ignore_bn=cfg.ignore_bn, ignore_bias=cfg.ignore_bias
        )
        torch.save((vector_map, coeff_dict), outputs_dir / "vector_map_and_coeffs.pt")

    # ===== STEP 03 =====
    # Calculate landscape limits
    coeffs = torch.cat(list(v - v[0] for v in coeff_dict.values()))
    max_x, max_y = coeffs.max(dim=0).values
    min_x, min_y = coeffs.min(dim=0).values
    logging.info("Coeff limits - X: [%f, %f], Y: [%f, %f]", min_x, max_x, min_y, max_y)
    extra_x = (max_x - min_x) * cfg.plot_extra
    extra_y = (max_y - min_y) * cfg.plot_extra
    x, y = np.meshgrid(
        np.linspace(min_x - extra_x, max_x + extra_x, cfg.plot_size),
        np.linspace(min_y - extra_y, max_y + extra_y, cfg.plot_size),
        indexing="ij",
    )
    logging.info("Landscape limits - X: [%f, %f], Y: [%f, %f]", x.min(), x.max(), y.min(), y.max())
    grid = torch.tensor(np.stack((x, y), axis=-1))
    logging.info("Landscape grid: %s", grid.shape)

    digits = int(math.log10(cfg.plot_size)) + 1  # For file naming purposes
    gpu_count = torch.cuda.device_count()
    max_workers = gpu_count if gpu_count > 0 else 1
    logging.info("Workers: %d (GPU count: %d)", max_workers, gpu_count)

    fit_job_queue = mp.Queue()
    result_queue = mp.Queue()
    for run_dir, ckpt_dir in end_point_ckpts.items():
        run_name = Path(run_dir).name
        config = NamedDict.from_yaml(Path(run_dir) / "config.yaml")
        for i, j in list(np.ndindex(grid.shape[:2])):
            log_name = f"{run_name}/x={i:0{digits}d}-y={j:0{digits}d}"
            if not (outputs_dir / log_name / "fit.done").exists():
                fit_job_queue.put(FitJob(
                    config=config,
                    ckpt_dir=ckpt_dir,
                    vector_map=vector_map,
                    target_coeffs=grid[i, j],
                    num_workers=cfg.dataset_workers,
                    ignore_bn=cfg.ignore_bn,
                    ignore_bias=cfg.ignore_bias,
                    outputs_dir=outputs_dir,
                    log_name=log_name,
                ))

    num_jobs = fit_job_queue.qsize()
    logging.info("Populated fit jobs: %d", num_jobs)
    if num_jobs:
        available_gpus = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
        if not available_gpus:
            logging.warning("No GPUs detected; falling back to CPU execution for fit jobs, which may be very slow.")
            available_gpus.append("cpu")
        processes = [
            mp.Process(target=worker, args=(worker_id, device, fit_job_queue, result_queue))
            for worker_id, device in enumerate(available_gpus[:max_workers])
        ]
        for p in processes:
            p.start()
        with tqdm.tqdm(total=num_jobs) as pbar:
            completed = 0
            while completed < num_jobs:
                result = result_queue.get()  # blocks until a worker finishes a job
                logging.info("Worker %d completed job %s with status: %s", result[0], result[1], result[2])
                completed += 1
                pbar.update(1)
                if result[2] == "failed":
                    logging.critical("A fit job failed; terminating all workers.")
                    for p in processes:
                        p.terminate()
                    return
        for p in processes:
            p.join()
    else:
        logging.info("No fit jobs to run; all donefiles already exist.")

    # ===== STEP 04 =====
    # Create plots
    pattern = r"/x=(\d+)-y=(\d+)/"
    for pm in cfg.plot_metrics:
        fig_name = f"{pm.name.replace('/', '-')}_{pm.mode}"
        result_grid_dict = dict()
        for run_dir in end_point_ckpts.keys():
            run_name = Path(run_dir).name
            result_grid_dict[run_name] = np.zeros((cfg.plot_size, cfg.plot_size))
            for path in (outputs_dir / run_name).glob("x=*-y=*/metrics.csv"):
                match = re.search(pattern, str(path))
                i, j = match.group(1), match.group(2)
                metrics_df = pd.read_csv(path).groupby("step", as_index=False).first()
                z = metrics_df.sort_values(
                    "progress/projection.l2"
                )[:cfg.plot_best_of_closest][pm.name]
                if pm.mode == "min":
                    z = z.min()
                elif pm.mode == "max":
                    z = z.max()
                else:
                    raise ValueError(f"Invalid mode value: {pm.mode}")
                logging.info("Extracted metric for %s at (%d, %d): %f", run_name, int(i), int(j), z)
                result_grid_dict[run_name][(int(i), int(j))] = z
        np.savez(outputs_dir / f"{fig_name}.npz", **result_grid_dict)
        logging.info("Result grids saved to %s", outputs_dir / f"{fig_name}.npz")

        if pm.log10:
            for key in result_grid_dict.keys():
                result_grid_dict[key] = np.log10(result_grid_dict[key])
        # global_min = min(v.min() for v in result_grid_dict.values())
        # global_max = max(v.max() for v in result_grid_dict.values())
        plot_trajectories = []
        for name, coeffs in coeff_dict.items():
            interp = RegularGridInterpolator(
                (x[:, 0], y[0, :]),
                result_grid_dict[name],
            )
            trajectory = coeffs - coeffs[0]  # Shift trajectory to start at (0, 0)
            plot_trajectories.append(
                go.Scatter3d(
                    x=trajectory[:, 0].numpy(),
                    y=trajectory[:, 1].numpy(),
                    z=interp(trajectory.numpy()),
                    mode="lines",
                    name=name,
                    line=dict(width=10, color=cfg.plot_colors[name][1]),
                )
            )
        plot_landscapes = [
            go.Surface(
                x=x,
                y=y,
                z=landscape,
                name=name,
                colorscale=cfg.plot_colors[name][0],
                opacity=cfg.plot_colors[name][2],
                showscale=False,
                showlegend=True,
                # cmin=global_min,
                # cmax=global_max,
            )
            for name, landscape in result_grid_dict.items()
        ]
        fig = go.Figure(data=plot_landscapes + plot_trajectories)
        fig.write_html(outputs_dir / f"{fig_name}.html")
        logging.info("Plot saved to %s", f"{fig_name}.html")


@dataclasses.dataclass
class FitJob:
    config: NamedDict
    ckpt_dir: Path
    vector_map: torch.Tensor
    target_coeffs: torch.Tensor
    num_workers: int
    ignore_bn: bool
    ignore_bias: bool
    outputs_dir: Path
    log_name: str


@dataclasses.dataclass
class FitResult:
    worker_id: int
    log_name: str
    status: str


def worker(worker_id: int, device: str, job_queue: mp.Queue, result_queue: mp.Queue):
    setup_logging_v2(Path("temp.log"))
    logging.info("Worker %d starting on device %s", worker_id, device)
    device = torch.device(device)
    dataset = None
    while True:
        try:
            try:
                job: FitJob = job_queue.get(timeout=5)  # Wait for a job
            except queue.Empty:
                logging.info("Worker %d found no more jobs and is exiting", worker_id)
                return

            donefile = job.outputs_dir / job.log_name / "fit.done"
            if donefile.exists():
                result_queue.put((worker_id, job.log_name, "skipped"))
                continue

            args = NamedDict(num_workers=job.num_workers)
            if dataset is None:
                dataset = getattr(datasets, job.config.dataset.type)(args, job.config.dataset)
                dataset.setup()
            task = getattr(tasks, job.config.task.type)(args, job.config.task, dataset.info)
            ckpt = torch.load(job.ckpt_dir, map_location=device, weights_only=False)
            state_dict = join_mask_and_weights(ckpt["state_dict"], task.state_dict())
            mask_as_state_dict(state_dict, task)
            task.load_state_dict(state_dict)

            projection_cb = GradualProjection(
                job.vector_map,
                job.target_coeffs,
                power=3.0,
                warmup_ratio=0.1,
                cooldown_ratio=0.1,
                ignore_bn=job.ignore_bn,
                ignore_bias=job.ignore_bias,
                correction_interval=10,
                driver="gelsd",
            )
            logger = pl.loggers.CSVLogger(
                save_dir=job.outputs_dir,
                name=job.log_name,
                version="",
            )
            job.config.trainer.log_every_n_steps = job.config.trainer.val_check_interval
            trainer = pl.Trainer(
                accelerator="gpu" if device.type == "cuda" else "cpu",
                devices=[int(str(device).split(":")[1])] if device.type == "cuda" else None,
                logger=[logger],
                callbacks=[projection_cb],
                enable_model_summary=False,
                enable_checkpointing=False,
                **job.config.trainer,
            )
            trainer.fit(task, datamodule=dataset)

            donefile.touch()
            result_queue.put((worker_id, job.log_name, "success"))

        except Exception as e:
            logging.critical("Worker %d encountered an error: %s", worker_id, str(e))
            logging.critical(traceback.format_exc())
            result_queue.put((worker_id, job.log_name, "failed"))
            return


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
