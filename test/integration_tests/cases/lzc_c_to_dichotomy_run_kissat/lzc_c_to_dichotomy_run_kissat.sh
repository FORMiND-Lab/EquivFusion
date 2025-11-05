#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

equiv_fusion -p "read_c -spec -top LZC LZC.mlir" \
             -p "read_v -impl -top lzc7_dichotomy lzc_dichotomy.v" \
             -p "equiv_miter -specModule LZC -implModule lzc7_dichotomy -mitermode aiger -o $OUTPUT_DIR/miter.aiger"

aigtocnf "$OUTPUT_DIR/miter.aiger" "$OUTPUT_DIR/miter.cnf"

#CHECK: UNSATISFIABLE
equiv_fusion -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.cnf"
