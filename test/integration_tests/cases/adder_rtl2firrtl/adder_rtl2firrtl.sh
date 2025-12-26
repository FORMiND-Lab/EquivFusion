#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_v -spec -top Adder Adder.sv" \
             -p "read_firrtl -impl -top Adder Adder.fir" \
             -p "equiv_miter -specModule Adder -implModule Adder -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver z3 --inputfile $OUTPUT_DIR/miter.smt"

#CHECK: unsat
equiv_fusion -p "read_v -spec -top Adder Adder.sv" \
             -p "read_firrtl -impl -top Adder Adder.fir" \
             -p "equiv_miter -specModule Adder -implModule Adder -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"


#CHECK: UNSATISFIABLE
equiv_fusion -p "read_v -spec -top Adder Adder.sv" \
             -p "read_firrtl -impl -top Adder Adder.fir" \
             -p "equiv_miter -specModule Adder -implModule Adder -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"

