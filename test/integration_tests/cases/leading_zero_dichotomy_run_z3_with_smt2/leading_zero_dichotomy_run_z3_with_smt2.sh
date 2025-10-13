#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

EXECUTE_SCRIPT="$CASE_DIR/../execute_solver_with_smt2.sh"

#CHECK: unsat
"$EXECUTE_SCRIPT" z3 "lzc7_dichotomy" "lzc7_dichotomy" "$OUTPUT_DIR" "$CASE_DIR/LZC_DICHOTOMY.mlir"

