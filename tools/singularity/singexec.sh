#!/usr/bin/env bash

set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
THIS_DIR=$ROOT_DIR/tools/singularity

if [ ! -z "$TMPDIR" ]; then
    extra_arg="-B $TMPDIR:$TMPDIR \
               -B $TMPDIR:/tmp"
else
    extra_arg="-B /tmp:/tmp"
fi

singexec () {
    singularity exec --containall \
        -B $ROOT_DIR:$ROOT_DIR \
        $extra_arg \
        $THIS_DIR/image.sif \
        bash -c ". /miniconda/etc/profile.d/conda.sh &&
            conda activate radiation &&
            cd $ROOT_DIR &&
            $1"
}
