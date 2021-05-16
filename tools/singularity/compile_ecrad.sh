#!/usr/bin/env bash

set -e
source tools/singularity/singexec.sh

if [ ! -d tools ]; then
    echo "Please run this script from the root folder"
    exit 1
fi

singexec "cd ecrad &&
    make clean &&
    make PROFILE=gfortran"
