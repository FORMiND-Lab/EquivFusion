#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

EXECUTE_SCRIPT="$CASE_DIR/../execute_solver_with_btor2.sh"

equivfusion-hls "$CASE_DIR/LZC.mlir" -o "$OUTPUT_DIR/LZC_hls.mlir"

#CHECK: unsat
"$EXECUTE_SCRIPT" bitwuzla "LZC" "lzc7_dichotomy" "$OUTPUT_DIR" "$OUTPUT_DIR/LZC_hls.mlir" "$CASE_DIR/lzc_dichotomy.mlir"
