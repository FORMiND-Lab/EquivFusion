#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_v -spec -top structure structure_assign.sv" \
             -p "read_v -impl -top structure structure_always.sv" \
             -p "equiv_miter -specModule structure -implModule structure -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver z3 --inputfile $OUTPUT_DIR/miter.smt"

#CHECK: unsat
equiv_fusion -p "read_v -spec -top structure structure_assign.sv" \
             -p "read_v -impl -top structure structure_always.sv" \
             -p "equiv_miter -specModule structure -implModule structure -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"


#CHECK: UNSATISFIABLE
equiv_fusion -p "read_v -spec -top structure structure_assign.sv" \
             -p "read_v -impl -top structure structure_always.sv" \
             -p "equiv_miter -specModule structure -implModule structure -mitermode aiger -o $OUTPUT_DIR/miter.aiger"
aigtocnf "$OUTPUT_DIR/miter.aiger" "$OUTPUT_DIR/miter.cnf"
equiv_fusion -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.cnf"

