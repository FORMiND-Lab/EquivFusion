#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}

#CHECK: unsat
equiv_fusion -p "read_v -spec test.v -top top" \
	     -p "read_v -impl -top top netlist.v ${CASE_DIR}/../../standard_library/cmos/cmos_cells.v" \
	     -p "equiv_miter --specModule top --implModule top --mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
	     -p "solver_runner --inputfile $OUTPUT_DIR/miter.btor2 --solver btormc" 


