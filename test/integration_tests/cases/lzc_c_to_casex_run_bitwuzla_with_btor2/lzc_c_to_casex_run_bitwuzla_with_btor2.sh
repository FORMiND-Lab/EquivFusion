#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_c -spec -top LZC LZC.mlir" \
             -p "read_v -impl -top lzc7_casex lzc_casex.v" \
             -p "equiv_miter -specModule LZC -implModule lzc7_casex -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"
