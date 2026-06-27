#!/usr/bin/env python3

import dataclasses
import glob
import logging
import queue
import sys
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import tqdm
from matplotlib import patheffects as pe
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
from torch import multiprocessing as mp

import datasets
import tasks
from gen_gsm_landscape import (
    extract_config_dir,
    sanity_check,
    step01_fit_pca,
    step02_build_vector_map,
)
from utils import NamedDict, setup_logging_v2
from utils.plot_utils import filter_named_parameters, flatten_model, prune_state_dict


@torch.no_grad()
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
        np.savez(  # savez_compressed sucks at nn weights
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

    available_gpus = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    max_workers = len(available_gpus) if available_gpus else 1
    logging.info("Workers: %d", max_workers)
    logging.info("Available GPUs: %s", available_gpus)

    landscape_results_path = outputs_dir / "landscape_results.npz"
    if landscape_results_path.exists():
        logging.info("Loading existing landscape results from %s", landscape_results_path)
        result_grid_dict = dict(np.load(landscape_results_path))
    else:
        result_grid_dict = {Path(run_dir).name: np.full(grid.shape[:2], np.nan) for run_dir in run_dirs}

    job_queue_list = []
    result_queue = mp.Queue()
    for run_dir, ckpt_dir in end_point_ckpts.items():
        run_name = Path(run_dir).name
        config = NamedDict.from_yaml(Path(run_dir) / "config.yaml")
        job_queue_list.append(mp.Queue())
        for i, j in np.ndindex(grid.shape[:2]):
            if not np.isnan(result_grid_dict[run_name][i, j]):
                continue  # Skip already computed points
            mult = grid[i, j] + coeff_dict[run_name][0]
            job = ValCoeffJob(
                run_name=run_name,
                i=i,
                j=j,
                config=config,
                ckpt_dir=ckpt_dir,
                mult=mult,
                vector_map=vector_map,
                dataset_workers=cfg.dataset_workers,
                ignore_bn=cfg.ignore_bn,
                ignore_bias=cfg.ignore_bias,
            )
            job_queue_list[-1].put(job)

    num_jobs = sum(q.qsize() for q in job_queue_list)
    logging.info("Total jobs queued: %d", num_jobs)

    if num_jobs:
        processes = [
            mp.Process(target=worker, args=(worker_id, device, job_queue_list, result_queue))
            for worker_id, device in enumerate(available_gpus[:max_workers])
        ]
        # while job_queue_list[-1].empty():
        #     pass  # Ensure at least one job is in the queue before starting workers
        for p in processes:
            p.start()
        try:
            with tqdm.tqdm(total=num_jobs) as pbar:
                completed = 0
                while completed < num_jobs:
                    result: ValCoeffResult = result_queue.get()  # blocks until a worker finishes a job
                    result_grid_dict[result.run_name][result.i, result.j] = result.loss
                    completed += 1
                    pbar.update(1)
        except KeyboardInterrupt as e:
            raise e
        finally:
            np.savez_compressed(outputs_dir / "landscape_results.npz", **result_grid_dict)
            logging.info("Results saved to %s", outputs_dir / "landscape_results.npz")
            for p in processes:
                p.join()
    else:
        logging.info("No jobs to process. All landscape points have already been computed.")

    # ===== STEP 04 =====
    # Create plots

    # -> Matplotlib plots
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "pgf.rcfonts": False,
    })
    plt.figure(figsize=(12, 10))

    lower_limit_grid = np.nanmin(list(result_grid_dict.values()), axis=0)
    vmin = np.floor(np.log10(np.nanmin(lower_limit_grid)))
    vmax = np.ceil(np.log10(np.nanmax(lower_limit_grid)))
    vmax = min(vmax, vmin + 3)  # Limit to 3 orders of magnitude
    for n in range(int(vmin), int(vmax) + 1):
        c = plt.contour(
            x, y, lower_limit_grid,
            levels=np.linspace(10 ** n, 10 ** (n + 1), num=9, endpoint=False),
            cmap="rainbow"
        )
        plt.clabel(c, inline=1, fontsize=8)

    colors = plt.cm.Set1(np.linspace(0, 1, len(coeff_dict)))
    for name, coeffs in coeff_dict.items():
        trajectory = coeffs - coeff_dict[name][0]  # Shift trajectory to start at (0, 0)
        line, = plt.plot(
            *zip(*trajectory),
            marker="o",
            ms=5,
            alpha=0.6,
            color=colors[list(coeff_dict.keys()).index(name)],
            label=name
        )
        line.set_path_effects([pe.Stroke(linewidth=2.2, foreground="black"), pe.Normal()])

    plt.xlabel(f"Explained Variance ratio: {pca.explained_variance_ratio_[0]:.3f}")
    plt.ylabel(f"Explained Variance ratio: {pca.explained_variance_ratio_[1]:.3f}")
    plt.title("Loss Landscape")
    plt.legend()

    plt.savefig(outputs_dir / "vis_landscape.pdf")
    logging.info("Plot saved to %s", outputs_dir / "vis_landscape.pdf")

    # -> Plotly plots
    if cfg.plotly_log10:
        for run_name in result_grid_dict:
            result_grid_dict[run_name] = np.log10(result_grid_dict[run_name])

    plot_trajectories = []
    for name, coeffs in coeff_dict.items():
        interp = RegularGridInterpolator(
            (x[:, 0], y[0, :]),
            result_grid_dict[name],
        )
        trajectory = coeffs - coeff_dict[name][0]  # Shift trajectory to start at (0, 0)
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
            showlegend=True,
            colorscale=cfg.plot_colors[name][0],
            opacity=cfg.plot_colors[name][2],
            showscale=False,
        )
        for name, landscape in result_grid_dict.items()
    ]
    fig = go.Figure(data=plot_landscapes + plot_trajectories)
    fig.write_html(outputs_dir / "vis_landscape.html")
    logging.info("Plot saved to %s", "vis_landscape.html")


@dataclasses.dataclass
class ValCoeffJob:
    run_name: str
    i: int
    j: int
    config: NamedDict
    ckpt_dir: Path
    mult: torch.Tensor
    vector_map: torch.Tensor
    dataset_workers: int
    ignore_bn: bool
    ignore_bias: bool


@dataclasses.dataclass
class ValCoeffResult:
    run_name: str
    i: int
    j: int
    loss: float


def worker(worker_id: int, device: str, job_queue_list: list[mp.Queue], result_queue: mp.Queue):
    # Uncomment the following line to enable logging within worker processes
    # setup_logging_v2(Path("temp.log"))
    logging.info("Worker %d starting on device %s", worker_id, device)
    new_queue_idx = None
    curr_queue_idx = None
    val_dataloader, task, final_weights = None, None, None
    with torch.no_grad():
        while True:
            if curr_queue_idx is None:
                # Previous queue finished, find the next queue
                if not job_queue_list:
                    logging.warning("Worker %d found no job queues available, exiting.", worker_id)
                    break
                new_queue_idx, longest_queue = max(
                    enumerate(job_queue_list), key=lambda q: q[1].qsize()
                )
                if longest_queue.empty():
                    logging.info("Worker %d found no jobs available, exiting.", worker_id)
                    break
                try:
                    job: ValCoeffJob = longest_queue.get_nowait()
                except queue.Empty:
                    # This can happen if another worker took the last job after we checked qsize
                    # Just loop again to find the next available job.
                    continue
            else:
                # Prioritize jobs from the same queue
                try:
                    job: ValCoeffJob = job_queue_list[curr_queue_idx].get_nowait()
                except queue.Empty:
                    # No more jobs for the current queue, look for the next queue
                    curr_queue_idx = None
                    continue
            logging.info(
                "Worker %d picked job for run %s at (%d, %d)",
                worker_id, job.run_name, job.i, job.j
            )

            # Initialization for the new queue if needed
            if curr_queue_idx is None:
                args = NamedDict(num_workers=job.dataset_workers)
                dataset = getattr(datasets, job.config.dataset.type)(args, job.config.dataset)
                dataset.setup(verbose=False)
                val_dataloader = dataset.val_dataloader()
                task = getattr(
                    tasks, job.config.task.type
                )(args, job.config.task, dataset.info, verbose=False)
                ckpt = torch.load(job.ckpt_dir, map_location=device, weights_only=False)
                state_dict = prune_state_dict(ckpt["state_dict"])
                task.load_state_dict(state_dict)
                task.model.to(device)
                final_weights = flatten_model(
                    task.model, consider_mask=True,
                    ignore_bn=job.ignore_bn,
                    ignore_bias=job.ignore_bias
                ).to(device)
                curr_queue_idx = new_queue_idx
            logging.info(
                "Worker %d initialized for run %s with checkpoint %s",
                worker_id, job.run_name, job.ckpt_dir
            )

            assert val_dataloader is not None
            assert task is not None
            assert final_weights is not None

            # Load new weights based on the job's vector map and multiplier
            mult = job.mult.to(device)
            vector_map = job.vector_map.to(device)
            mapped_weights = final_weights + vector_map.T @ mult
            weight_index = 0
            for _name, param in filter_named_parameters(
                task.model, ignore_bn=job.ignore_bn, ignore_bias=job.ignore_bias
            ):
                param.copy_(mapped_weights[weight_index:weight_index + param.numel()].view_as(param))
                weight_index += param.numel()

            # Evaluate the model on the validation set
            task.eval()
            total_loss, total_samples = 0, 0
            for batch in val_dataloader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                pred = task(x)
                loss = task.loss_fn(pred, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
            result = ValCoeffResult(
                run_name=job.run_name,
                i=job.i,
                j=job.j,
                loss=total_loss / total_samples,
            )
            result_queue.put(result)
            logging.info(
                "Worker %d completed job for run %s at (%d, %d) with loss %.4f",
                worker_id, job.run_name, job.i, job.j, result.loss
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
