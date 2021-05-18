#!/usr/bin/env bash

#PBS -lwalltime=1:00:00
#PBS -lselect=1:ncpus=1:mem=8gb

#PBS -o logs/
#PBS -e logs/

set -ex

JOB_NAME=stats

hostname
printenv
date
lscpu
singularity --version

ROOT_DIR=$PBS_O_WORKDIR/..
NB_DIR=$ROOT_DIR/notebooks
RESULTS_DIR=$ROOT_DIR/results
RESULTS_NOTEBOOK=$RESULTS_DIR/notebooks

SIF_PATH=$ROOT_DIR/tools/singularity/image.sif

singularity exec --containall \
  -B $ROOT_DIR:$ROOT_DIR \
  -B $TMPDIR:/tmp \
  $SIF_PATH \
  bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate radiation && cd $NB_DIR && \
    PYTHONHASHSEED=0 jupyter nbconvert --to html --execute grid-search-stats.ipynb \
    --output=$RESULTS_NOTEBOOK/stats \
    --ExecutePreprocessor.timeout=86400 --ExecutePreprocessor.iopub_timeout=300"