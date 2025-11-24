#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_v -spec -top dff dff.v" \
             -p "unroll --spec --steps 3" \
             -p "read_v -impl -top dff dff.v" \
             -p "unroll --impl --steps 3" \
             -p "equiv_miter -specModule dff -implModule dff -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver z3 --inputfile $OUTPUT_DIR/miter.smt"
