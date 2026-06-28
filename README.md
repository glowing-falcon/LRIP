# Loss Landscape-Guided Pruning with Integrated Learning Rate Schedules for Efficient Training-Time Network Compression

This repository contains the code and configurations for experiments conducted in the main paper of "Loss Landscape-Guided Pruning with Integrated Learning Rate Schedules for Efficient Training-Time Network Compression" (ACCV 2026).

## Prequisites

uv is recommended for environment setup. `uv.lock` file is included in order to reproduce the environment used in the experiments. 

The datasets used in the paper are not included in this repository. Please download them and place them in the `data/` directory:

```
.
|-- CIFAR10
|   |-- cifar-10-batches-py
|   `-- cifar-10-python.tar.gz
|-- OxfordIIITPet
|   `-- oxford-iiit-pet
|       |-- annotations
|       `-- images
|-- StanfordCars
|   `-- stanford_cars
|       |-- cars_test
|       |-- cars_test_annos_withlabels.mat
|       |-- cars_train
|       `-- devkit
`-- imagenet
    |-- ILSVRC2012_devkit_t12
    |-- src
    |-- train
    `-- val
```

Although not required, GPU is recommended for training and evaluation. The code has been tested on Ubuntu 24.04 with cuda 12.8

## Setup

Install the dependencies using the following command:

```bash
make install
```

Our training configuration files originally provided in the `exp/` directory are used to generate the actual configuration files used for training. To systematically generate the training configuration files for `exp/exp10/agp` for example, run the following command:

```bash
./src/gen_combinations.py \
    --infile exp/exp10/agp/combinations.yaml \
    --outdir exp/exp10/agp/configs
```

This will generate the training configuration files in `exp/exp10/agp/configs` based on the combinations specified in `exp/exp10/agp/combinations.yaml`. The same command can be used for other experiments by changing the `--infile` and `--outdir` arguments accordingly.

Training can be performed using the following command, i.e. for the first experiment in `exp/exp10/agp`:

```bash
./src/main.py \
    --config "exp/exp10/agp/configs/ds=cif10-model=effb3-seed=s1.yaml" \
    --runs-dir "runs/exp10/agp" \
    --run-name "ds=cif10-model=effb3-seed=s1" \
    --num-workers 4 
```

Our GSM plotting configurations are found in the `plots/` directory. The plotting configurations are used to generate the points for the GSM plots in the main paper. When generating the GSM plots, ensure that the experiment counterpart has already been run and the checkpoints are available in the `runs/` directory. For example, to generate the GSM plot for `exp/vis03d`, run the following command:

```bash
./src/gen_gsm_landscape.py \
    --config-dir "plots/vis03/resnet-50"
```

## Experiments

Since some of the experiments overlap between tables and figures, we reuse the same configuration and results for these experiments. The following sections describe the configuration locations for each table and figure in the main paper.

### Table 1

Note:

* All LR in `exp/exp10/*` is `1e-4` with the exception of `exp/exp10/norpune_lrfix`
* "Lesser datasets" refer to CIFAR 10, Oxford Pets and Stanford Cars

Configuration locations:

* `exp/exp10/agp` configs used for the "GMP" column for the lesser datasets
* `exp/exp10/lrip` configs used for the "LRIP" column for the lesser datasets
* `exp/exp10/lth` configs used for the "LTH" column
* `exp/exp10/lrr` configs used for the "LRR" column
* `exp/exp10/noprune` configs used for the "No pruning" column for the lesser datasets
* `exp/exp10/noprune_lrfix` configs used for the "No pruning" column for ImageNet
* `exp/exp16/agp` configs used for "GMP" on ImageNet
* `exp/exp16/lrip` configs used for "LRIP" on ImageNet

### Table 2

Configuration locations:

* `exp/exp13/lrip-cyclical` configs used for "Cyclical"
* `exp/exp13/lrip-force` configs used for "FORCE"
* `exp/exp13/lrip-forcemag` configs used for "FORCE Magnitude"
* `exp/exp13/lrip-wanda` configs used for "Wanda"

### Table 3

Configuration locations:

* `exp/exp16/agp` configs used for all "GMP" rows
* `exp/exp16/lrip` configs used for all "LRIP" rows

### Table 4

Configuration locations:

* `exp/exp14` configs used for all p values

### Table 5

Configuration locations:

* `exp/exp15/agp` configs used for all "GMP" rows
* `exp/exp15/lrip` configs used for all "LRIP" rows

### Figure 3

Configuration locations:

* `exp/exp11/agp` configs for "vitbs GMP" with sparsities > 0
* `exp/exp11/lrip` configs for "vitbs LRIP" with sparsities > 0
* `exp/exp11a/agp` configs for "res50 GMP" and "effb3 GMP" with sparsities > 0
* `exp/exp11a/lrip` configs for "res50 LRIP" and "effb3 LRIP" with sparsities > 0
* `exp/exp10/noprune` configs for "vitbs GMP" and "vitbs LRIP" with sparsity = 0 
* `exp/exp10/norpune_lrfix` configs for "res50 GMP", "res50 LRIP", "effb3 GMP" and "effb3 LRIP" with sparsity = 0

### Figure 4 (a) and (b)

Configuration locations:

* `exp/vis03d` configs to perform training as described in Section 4.3 step 1

Plotting configurations:

* `plots/vis03/resnet-50` config to generate plotting points for all landscapes in Figure 4 (a) and (b)

### Figure 4 (c) and (d)

Configuration locations:

* `exp/vis03e` configs to perform training as described in Section 4.3 step 1

Plotting configurations:

* `plots/vis03/vit-small` config to generate plotting points for all landscapes in Figure 4 (c) and (d)

