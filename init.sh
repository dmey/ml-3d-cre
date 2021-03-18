#!/usr/bin/env bash

set -e

echo "Extracting all compressed NetCDF files..."
find -name "*.nc.xz" -exec unxz -k '{}' \;

echo "Done"
