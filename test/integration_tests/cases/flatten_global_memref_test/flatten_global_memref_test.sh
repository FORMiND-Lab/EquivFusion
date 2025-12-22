#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}


#CHECK-NOT: memref<2x2xi32>
equivfusion-opt "test.mlir" --equivfusion-flatten-memref


