#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}


#CHECK-NOT: memref.get_global
equivfusion-opt "fullyConnected_affine.mlir" --equivfusion-get-global-to-alloc


