#!/usr/bin/env bash

set -xe

ROOT_DIR=$(git rev-parse --show-toplevel)
THIS_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

echo "ROOT_DIR=$ROOT_DIR"

singularity exec --containall \
    -B $ROOT_DIR:$ROOT_DIR \
    $THIS_DIR/image.sif \
    bash -c "cd $ROOT_DIR/ecrad && make clean && make PROFILE=gfortran"
