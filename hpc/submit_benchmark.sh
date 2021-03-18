#!/usr/bin/env bash

#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=48:mem=124gb

#PBS -o logs/
#PBS -e logs/

set -ex

# Adjust this to the available cores in the job
NCORES=48
GPU=0
# Adjust this to the model configuration to benchmark
CASE_NAME=split
VAR_SYNTH=sw_albedo,cos_solar_zenith_angle
SYNTH_MUL_FACTOR=0
UNIF_RATIO=0.0
STRETCH_FACTOR=1.0
MODEL_TYPE=MLP
N_HIDDEN_LAYERS=3
HIDDEN_SIZE=1
LOSS=mse
ACTIVATION=elu
L1_PENALTY=1e-5
L2_PENALTY=1e-5
VAR_REGULARIZER_FACTOR=0
DROPOUT_RATIO_INPUT=0
DROPOUT_RATIO_HIDDEN=0
VAR_ML=optical_depth_fl,cos_solar_zenith_angle,sw_albedo,skin_temperature,cloud_fraction,temperature_fl
EPOCHS=1
SAVE_MODEL=1
ITERATION=0
JOB_NAME=benchmark

hostname
printenv
date
lscpu
singularity --version

ROOT_DIR=$PBS_O_WORKDIR/..
NB_DIR=$ROOT_DIR/notebooks
RESULTS_DIR=$ROOT_DIR/results

SIF_PATH=$ROOT_DIR/tools/singularity/image.sif

rm -rf $RESULTS_DIR/job_stats_$JOB_NAME
mkdir -p $RESULTS_DIR

# generate model and results file for benchmark script
export SINGULARITYENV_PBS_JOBID=$PBS_JOBID
export SINGULARITYENV_JOB_NAME=$JOB_NAME
export SINGULARITYENV_CASE_NAME=$CASE_NAME
export SINGULARITYENV_EPOCHS=$EPOCHS
export SINGULARITYENV_SAVE_MODEL=$SAVE_MODEL
export SINGULARITYENV_ITERATION=$ITERATION
export SINGULARITYENV_SYNTH_MUL_FACTOR=$SYNTH_MUL_FACTOR
export SINGULARITYENV_VAR_SYNTH=$VAR_SYNTH
export SINGULARITYENV_UNIF_RATIO=$UNIF_RATIO
export SINGULARITYENV_STRETCH_FACTOR=$STRETCH_FACTOR
export SINGULARITYENV_MODEL_TYPE=$MODEL_TYPE
export SINGULARITYENV_VAR_ML=$VAR_ML
export SINGULARITYENV_LOSS=$LOSS
export SINGULARITYENV_ACTIVATION=$ACTIVATION
export SINGULARITYENV_L1_PENALTY=$L1_PENALTY
export SINGULARITYENV_L2_PENALTY=$L2_PENALTY
export SINGULARITYENV_VAR_REGULARIZER_FACTOR=$VAR_REGULARIZER_FACTOR
export SINGULARITYENV_DROPOUT_RATIO_INPUT=$DROPOUT_RATIO_INPUT
export SINGULARITYENV_DROPOUT_RATIO_HIDDEN=$DROPOUT_RATIO_HIDDEN
export SINGULARITYENV_N_HIDDEN_LAYERS=$N_HIDDEN_LAYERS
export SINGULARITYENV_HIDDEN_SIZE=$HIDDEN_SIZE

singularity exec --containall \
  -B $ROOT_DIR:$ROOT_DIR \
  -B $TMPDIR:/tmp \
  $SIF_PATH \
  bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate radiation && cd $NB_DIR && \
    PYTHONHASHSEED=0 jupyter nbconvert --to html --execute ml.ipynb \
    --ExecutePreprocessor.timeout=86400 --ExecutePreprocessor.iopub_timeout=300"

# run benchmarks
export SINGULARITYENV_NCORES=$NCORES
export SINGULARITYENV_GPU=$GPU

singularity exec --containall \
  -B $ROOT_DIR:$ROOT_DIR \
  -B $TMPDIR:/tmp \
  $SIF_PATH \
  bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate radiation && cd $ROOT_DIR && \
    python tools/run_benchmarks.py > $RESULTS_DIR/benchmark_GPU=$GPU.txt"
