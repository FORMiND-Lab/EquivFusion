#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

EXECUTE_SCRIPT="$CASE_DIR/../execute_solver_with_btor2.sh"

#CHECK: unsat
"$EXECUTE_SCRIPT" bitwuzla "$CASE_DIR/top.mlir" "$OUTPUT_DIR"
