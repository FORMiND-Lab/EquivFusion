#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

EXECUTE_SCRIPT="$CASE_DIR/../execute_solver_with_cnf.sh"

#CHECK: UNSATISFIABLE
"$EXECUTE_SCRIPT" kissat "lzc7_loop" "lzc7_loop" "$OUTPUT_DIR" "$CASE_DIR/LZC_LOOP.mlir"

