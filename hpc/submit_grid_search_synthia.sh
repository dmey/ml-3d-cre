#!/usr/bin/env bash

#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=32:mem=62gb
#PBS -J 1-10

#PBS -o logs/
#PBS -e logs/

if [[ "$DRY_RUN" == "" ]]; then

set -ex

# Adjust this to the available cores in the job
NCORES=32

hostname
printenv
date
lscpu
singularity --version

ROOT_DIR=$PBS_O_WORKDIR/..
NB_DIR=$ROOT_DIR/notebooks
RESULTS_DIR=$ROOT_DIR/results
SIF_PATH=$ROOT_DIR/tools/singularity/image.sif

RESULTS_NOTEBOOK=$RESULTS_DIR/notebooks/$JOB_NAME

mkdir -p $RESULTS_NOTEBOOK

fi

# Constants
EPOCHS=1000
SAVE_MODEL=0 # only needed for benchmarks, see submit_benchmark.sh

# Defaults for all job types (can be overridden below)
CASE_NAME_VALS=( split ) # combined
ITERATION_VALS=( $(seq 0 9 ) )
VAR_SYNTH_VALS=( sw_albedo,cos_solar_zenith_angle )
USE_DIFF_VALS=( 0 1 )
USE_HEATING_RATES_VALS=( 1 )

case $JOB_NAME in


mlp_synthia)
  USE_DIFF_VALS=( 1 )
  ITERATION_VALS=( $(seq 0 9 ) )
  STORE_HTML=1
  COPULA_TYPE_VALS=( gaussian )
  SYNTH_MUL_FACTOR_VALS=( 0 9 )
  VAR_SYNTH_VALS=( 
    sw_albedo,cos_solar_zenith_angle
    )
  UNIF_RATIO_VALS=( 0 )
  STRETCH_FACTOR_VALS=( 1.0 )
  MODEL_TYPE_VALS=( MLP )
  RNN_TYPE_VALS=( GRU )
  RNN_DIRECTION_VALS=( bi )
  N_HIDDEN_LAYERS_VALS=( 3 )
  HIDDEN_SIZE_VALS=( 1 )
  LOSS_VALS=( mse )
  ACTIVATION_VALS=( elu )
  L1_PENALTY_VALS=( 1e-5 )
  L2_PENALTY_VALS=( 1e-5 )
  VAR_REGULARIZER_FACTOR_VALS=( 0 )
  DROPOUT_RATIO_INPUT_VALS=( 0 )
  DROPOUT_RATIO_HIDDEN_VALS=( 0 )
  VAR_ML_VALS=(
    cos_solar_zenith_angle,sw_albedo,skin_temperature,optical_depth_fl,cloud_fraction,temperature_fl
  )
  ;;


*)
  echo "Unknown JOB_NAME=$JOB_NAME"
  exit 1
  ;;
esac

i=0

for ITERATION in ${ITERATION_VALS[@]}; do
for CASE_NAME in ${CASE_NAME_VALS[@]}; do
for USE_DIFF in ${USE_DIFF_VALS[@]}; do
for USE_HEATING_RATES in ${USE_HEATING_RATES_VALS[@]}; do
for COPULA_TYPE in ${COPULA_TYPE_VALS[@]}; do
for SYNTH_MUL_FACTOR in ${SYNTH_MUL_FACTOR_VALS[@]}; do
for VAR_SYNTH in ${VAR_SYNTH_VALS[@]}; do
for UNIF_RATIO in ${UNIF_RATIO_VALS[@]}; do
for STRETCH_FACTOR in ${STRETCH_FACTOR_VALS[@]}; do
for VAR_ML in ${VAR_ML_VALS[@]}; do
for LOSS in ${LOSS_VALS[@]}; do
for ACTIVATION in ${ACTIVATION_VALS[@]}; do
for L1_PENALTY in ${L1_PENALTY_VALS[@]}; do
for L2_PENALTY in ${L2_PENALTY_VALS[@]}; do
for VAR_REGULARIZER_FACTOR in ${VAR_REGULARIZER_FACTOR_VALS[@]}; do
for DROPOUT_RATIO_INPUT in ${DROPOUT_RATIO_INPUT_VALS[@]}; do
for DROPOUT_RATIO_HIDDEN in ${DROPOUT_RATIO_HIDDEN_VALS[@]}; do
for MODEL_TYPE in ${MODEL_TYPE_VALS[@]}; do
for RNN_TYPE in ${RNN_TYPE_VALS[@]}; do
for RNN_DIRECTION in ${RNN_DIRECTION_VALS[@]}; do
for N_HIDDEN_LAYERS in ${N_HIDDEN_LAYERS_VALS[@]}; do
for HIDDEN_SIZE in ${HIDDEN_SIZE_VALS[@]}; do

  ((i=i+1))
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  if [[ $i -ne $PBS_ARRAY_INDEX ]]; then
    continue
  fi

  export SINGULARITYENV_PBS_JOBID=$PBS_JOBID
  export SINGULARITYENV_PBS_ARRAY_INDEX=$PBS_ARRAY_INDEX
  export SINGULARITYENV_JOB_NAME=$JOB_NAME
  export SINGULARITYENV_CASE_NAME=$CASE_NAME
  export SINGULARITYENV_NCORES=$NCORES
  export SINGULARITYENV_EPOCHS=$EPOCHS
  export SINGULARITYENV_SAVE_MODEL=$SAVE_MODEL
  export SINGULARITYENV_ITERATION=$ITERATION
  export SINGULARITYENV_USE_DIFF=$USE_DIFF
  export SINGULARITYENV_USE_HEATING_RATES=$USE_HEATING_RATES
  export SINGULARITYENV_COPULA_TYPE=$COPULA_TYPE
  export SINGULARITYENV_SYNTH_MUL_FACTOR=$SYNTH_MUL_FACTOR
  export SINGULARITYENV_VAR_SYNTH=$VAR_SYNTH
  export SINGULARITYENV_UNIF_RATIO=$UNIF_RATIO
  export SINGULARITYENV_STRETCH_FACTOR=$STRETCH_FACTOR
  export SINGULARITYENV_VAR_ML=$VAR_ML
  export SINGULARITYENV_LOSS=$LOSS
  export SINGULARITYENV_ACTIVATION=$ACTIVATION
  export SINGULARITYENV_L1_PENALTY=$L1_PENALTY
  export SINGULARITYENV_L2_PENALTY=$L2_PENALTY
  export SINGULARITYENV_VAR_REGULARIZER_FACTOR=$VAR_REGULARIZER_FACTOR
  export SINGULARITYENV_DROPOUT_RATIO_INPUT=$DROPOUT_RATIO_INPUT
  export SINGULARITYENV_DROPOUT_RATIO_HIDDEN=$DROPOUT_RATIO_HIDDEN
  export SINGULARITYENV_MODEL_TYPE=$MODEL_TYPE
  export SINGULARITYENV_RNN_TYPE=$RNN_TYPE
  export SINGULARITYENV_RNN_DIRECTION=$RNN_DIRECTION
  export SINGULARITYENV_N_HIDDEN_LAYERS=$N_HIDDEN_LAYERS
  export SINGULARITYENV_HIDDEN_SIZE=$HIDDEN_SIZE

  if [[ "$STORE_HTML" == "1" ]]; then
    filename="case=$CASE_NAME,use_diff=$USE_DIFF,use_heating_rates=$USE_HEATING_RATES,copula_type=$COPULA_TYPE,synth_mul_factor=$SYNTH_MUL_FACTOR,unif_ratio=$UNIF_RATIO,stretch_factor=$STRETCH_FACTOR,var_synth=$VAR_SYNTH,var_ml=$VAR_ML,loss=$LOSS,activation=$ACTIVATION,l1_penalty=$L1_PENALTY,l2_penalty=$L2_PENALTY,var_regularizer_factor=$VAR_REGULARIZER_FACTOR,dropout_ratio_input=$DROPOUT_RATIO_INPUT,dropout_ratio_hidden=$DROPOUT_RATIO_HIDDEN,model_type=$MODEL_TYPE,rnn_type=$RNN_TYPE,rnn_direction=$RNN_DIRECTION,layers=$N_HIDDEN_LAYERS,size=$HIDDEN_SIZE,iteration=$ITERATION"
    # hash filename to avoid exceeding max filename length (255)
    hashed_filename=$(echo -n "$filename" | md5sum | cut -d ' ' -f 1)
    echo "$filename" > $RESULTS_NOTEBOOK/$hashed_filename.txt
    out_arg="--output=$RESULTS_NOTEBOOK/$hashed_filename"
  else
    out_arg="--output-dir=$TMPDIR"
  fi

  singularity exec --containall \
    -B $ROOT_DIR:$ROOT_DIR \
    -B $TMPDIR:/tmp \
    $SIF_PATH \
    bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate radiation && cd $NB_DIR && \
      PYTHONHASHSEED=0 jupyter nbconvert --to html --execute ml.ipynb \
      $out_arg \
      --ExecutePreprocessor.timeout=86400 --ExecutePreprocessor.iopub_timeout=300"

  exit 0
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "#PBS -J 1-$i"
fi