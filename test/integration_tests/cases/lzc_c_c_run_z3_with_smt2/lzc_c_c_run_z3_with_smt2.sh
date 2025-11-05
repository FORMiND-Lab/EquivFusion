#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_c -spec -top LZC LZC.mlir" \
             -p "read_c -impl -top LZC LZC.mlir" \
             -p "equiv_miter -specModule LZC -implModule LZC -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver z3 --inputfile $OUTPUT_DIR/miter.smt"
