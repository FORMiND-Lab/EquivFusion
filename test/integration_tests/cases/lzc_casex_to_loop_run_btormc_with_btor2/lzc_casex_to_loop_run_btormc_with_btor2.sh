#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_v -spec -top lzc7_casex lzc_casex.v" \
             -p "read_v -impl -top lzc7_loop lzc_loop.v" \
             -p "equiv_miter -specModule lzc7_casex -implModule lzc7_loop -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver btormc --inputfile $OUTPUT_DIR/miter.btor2"
