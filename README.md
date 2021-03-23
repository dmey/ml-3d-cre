## Overview

This is the development repository for the paper "[Machine Learning Emulation of 3D Cloud Radiative Effects](https://arxiv.org/abs/2103.11919)" simply meant as a placeholder.

**For the paper's data archive that includes all model outputs and the [Singularity](https://sylabs.io/singularity/) image, please refer to the [archive on Zenodo](https://doi.org/10.5281/zenodo.4625414) instead**.

For the Python tool to generate the synthetic data, please refer to the [Synthia repository](https://github.com/dmey/synthia).


## Requirements

- Linux
- [Singularity](https://sylabs.io/singularity/) >= 3
- [Portable Batch System](https://en.wikipedia.org/wiki/Portable_Batch_System) (PBS) job scheduler*

*Although PBS in not a strict requirement, it is required to run all helper scripts as included in this repository. Please note that depending on your specific system settings and resource availability, you may need to modify PBS parameters at the top of submit scripts stored in the `hpc` directory (e.g. `#PBS -lwalltime=24:00:00`).


## Initialization

Deflate the data archive with:

```
./init.sh
```

Build the Singularity image with:

```
singularity build --remote tools/singularity/image.sif tools/singularity/image.def
```

Compile ecRad with Singularity:

```
./tools/singularity/compile_ecrad.sh
```

## Usage

To reproduce the results as described in the paper, run the following commands from the `hpc` folder:

```
qsub -v JOB_NAME=mlp_synthia ./submit_grid_search_synthia.sh
qsub -v JOB_NAME=mlp_default ./submit_grid_search_default.sh
qsub submit_benchmark.sh
```

then, to plot stats and identify notebooks run:

```
qsub submit_stats.sh
```

## Local development

For local development, notebooks can be run independently. To install the required dependencies, run the following through [Anaconda](https://docs.conda.io/en/latest/miniconda.html) `conda env create -f environment.yml`. Then, to activate the environment use `conda activate radiation`. For ecRad, the list of system dependencies are listed in `tools\singularity\image.def` and can be run with `tools\singularity\compile_ecrad.sh`.


## Licence

Paper code released under the [MIT license](./LICENSE.txt). Data released under [CC BY 4.0](./data/LICENSE.txt). [ecRad](https://confluence.ecmwf.int/display/ECRAD) released under the [Apache 2.0 license](./ecrad/LICENSE).
