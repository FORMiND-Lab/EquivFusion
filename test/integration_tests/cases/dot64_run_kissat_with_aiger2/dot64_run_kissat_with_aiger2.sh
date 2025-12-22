#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: UNSATISFIABLE
equiv_fusion -p "read_c -spec -top mm mm.mlir" \
             -p "read_v -impl -top dot64_comb dot64.v" \
             -p "equiv_miter -specModule mm -implModule dot64_comb -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"
