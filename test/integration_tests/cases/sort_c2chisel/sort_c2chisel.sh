#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}


echo "====================================================================== Equal ========================================================================"

echo "--------------------------------------------------------------- Run bitwuzla with smt ---------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}unsat{{[[:space:]]|$}}
equiv_fusion -p "set_port -output output" \
             -p "read_c -spec -top Sort Sort.mlir" \
             -p "read_firrtl -impl -top Sort  Sort.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Sort -implModule Sort -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.smt"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run bitwuzla with btor2 -------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}unsatisfiable.{{[[:space:]]|$}}
equiv_fusion -p "set_port -output output" \
             -p "read_c -spec -top Sort Sort.mlir" \
             -p "read_firrtl -impl -top Sort  Sort.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Sort -implModule Sort -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run kissat  -------------------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}UNSATISFIABLE{{[[:space:]]|$}}
equiv_fusion -p "set_port -output output" \
             -p "read_c -spec -top Sort Sort.mlir" \
             -p "read_firrtl -impl -top Sort  Sort.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Sort -implModule Sort -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"
echo -e "\n\n"



echo "====================================================================== Unequal ======================================================================"


echo "--------------------------------------------------------------- Run bitwuzla with smt ---------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}sat{{[[:space:]]|$}}
equiv_fusion -p "set_port -output output" \
             -p "read_c -spec -top Sort Sort_Unequal.mlir" \
             -p "read_firrtl -impl -top Sort Sort.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Sort -implModule Sort -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.smt"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run bitwuzla with btor2 -------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}satisfiable.{{[[:space:]]|$}}
equiv_fusion -p "set_port -output output" \
             -p "read_c -spec -top Sort Sort_Unequal.mlir" \
             -p "read_firrtl -impl -top Sort  Sort.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Sort -implModule Sort -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run kissat  -------------------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}SATISFIABLE{{[[:space:]]|$}}
equiv_fusion -p "set_port -output output" \
             -p "read_c -spec -top Sort Sort_Unequal.mlir" \
             -p "read_firrtl -impl -top Sort  Sort.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Sort -implModule Sort -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"
echo -e "\n\n"
