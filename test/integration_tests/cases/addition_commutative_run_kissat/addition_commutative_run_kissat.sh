#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

EXECUTE_SCRIPT="$CASE_DIR/../execute_solver_with_cnf.sh"

#CHECK: UNSATISFIABLE
"$EXECUTE_SCRIPT" kissat "$CASE_DIR/foo1.mlir"  "$CASE_DIR/foo2.mlir" "$OUTPUT_DIR" "aiger"
