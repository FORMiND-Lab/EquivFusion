#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: UNSATISFIABLE
equiv_fusion -p "read_c -spec -top test multiple_dimensional_test.mlir" \
             -p "read_c -impl -top test multiple_dimensional_test.mlir" \
             -p "equiv_miter -specModule test -implModule test -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"

