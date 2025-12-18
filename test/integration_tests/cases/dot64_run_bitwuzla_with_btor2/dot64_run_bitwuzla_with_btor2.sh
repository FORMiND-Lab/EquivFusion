#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_c -spec -top mm mm.mlir" \
             -p "read_v -impl -top dot64_comb dot64.v" \
             -p "equiv_miter -specModule mm -implModule dot64_comb -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"
