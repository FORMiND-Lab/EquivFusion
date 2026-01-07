#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}


echo "====================================================================== Equal ========================================================================"

echo "--------------------------------------------------------------- Run bitwuzla with smt ---------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}unsat{{[[:space:]]|$}}
equiv_fusion -p "read_c -spec -top Dot64 Dot64.mlir" \
             -p "read_firrtl -impl -top Dot64 Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.smt"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run bitwuzla with btor2 -------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}unsatisfiable.{{[[:space:]]|$}}
equiv_fusion -p "read_c -spec -top Dot64 Dot64.mlir" \
             -p "read_firrtl -impl -top Dot64  Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run kissat  -------------------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}UNSATISFIABLE{{[[:space:]]|$}}
equiv_fusion -p "read_c -spec -top Dot64 Dot64.mlir" \
             -p "read_firrtl -impl -top Dot64  Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"
echo -e "\n\n"



echo "====================================================================== Unequal ======================================================================"


echo "--------------------------------------------------------------- Run bitwuzla with smt ---------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}sat{{[[:space:]]|$}}
equiv_fusion -p "read_c -spec -top Dot64 Dot64_Unequal.mlir" \
             -p "read_firrtl -impl -top Dot64 Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode smtlib -o $OUTPUT_DIR/miter.smt" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.smt"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run bitwuzla with btor2 -------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}satisfiable.{{[[:space:]]|$}}
equiv_fusion -p "read_c -spec -top Dot64 Dot64_Unequal.mlir" \
             -p "read_firrtl -impl -top Dot64  Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode btor2 -o $OUTPUT_DIR/miter.btor2" \
             -p "solver_runner --solver bitwuzla --inputfile $OUTPUT_DIR/miter.btor2"
echo -e "\n\n"


echo "--------------------------------------------------------------- Run kissat  -------------------------------------------------------------------------"
#CHECK: {{[[:space:]]|^}}SATISFIABLE{{[[:space:]]|$}}
equiv_fusion -p "read_c -spec -top Dot64 Dot64_Unequal.mlir" \
             -p "read_firrtl -impl -top Dot64  Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
             -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode aiger -o $OUTPUT_DIR/miter.aiger" \
             -p "solver_runner --solver kissat --inputfile $OUTPUT_DIR/miter.aiger"
echo -e "\n\n"