#!/usr/bin/env bash

set -e
source tools/singularity/singexec.sh

if [ ! -d tools ]; then
    echo "Please run this script from the root folder"
    exit 1
fi

nml_path=$1
in_path=$2
out_path=$3

singexec "export OMP_NUM_THREADS= &&
    cd ecrad &&
    bin/ecrad $nml_path $in_path $out_path"
