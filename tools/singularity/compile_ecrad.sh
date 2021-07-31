#!/usr/bin/env bash

set -xe

# https://stackoverflow.com/a/246128
local THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
local ROOT_DIR=$THIS_DIR/../..

echo "ROOT_DIR=$ROOT_DIR"

singularity exec --containall \
    -B $ROOT_DIR:$ROOT_DIR \
    $THIS_DIR/image.sif \
    bash -c "cd $ROOT_DIR/ecrad && make clean && make PROFILE=gfortran"
