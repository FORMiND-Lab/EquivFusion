#!/usr/bin/bash

set -euo pipefail

CASE_DIR=$(dirname "$(realpath "$0")")
OUTPUT_DIR=${CASE_LOG_PATH:-.}


run_equivalence_test() {
    local impl_file="$1"
    local miter_mode="$2"
    local solver_name="$3"

    local output_suffix
    case "$miter_mode" in
        smtlib) output_suffix="smt" ;;
        btor2) output_suffix="btor2" ;;
        aiger) output_suffix="aiger" ;;
       *)
            echo "错误: 未知的 mitermode: $miter_mode"
            return 1
            ;;
    esac
    local output_file="$OUTPUT_DIR/miter.$output_suffix"

    echo "--------------------------------------------------------------- Run $solver_name with $output_suffix"

    equiv_fusion -p "read_firrtl -spec -top Dot64 Dot64.fir --scalarize-public-modules false --scalarize-public-modules false --preserve-aggregate all" \
                 -p "read_c -impl -top Dot64 $impl_file" \
                 -p "equiv_miter -specModule Dot64 -implModule Dot64 -mitermode $miter_mode -o $output_file" \
                 -p "solver_runner --solver $solver_name --inputfile $output_file" \
                 -p "show"

    echo -e "\n\n"
}


echo "====================================================================== Equal ========================================================================"

#CHECK: {{[[:space:]]|^}}unsat{{[[:space:]]|$}}
run_equivalence_test "Dot64.mlir", "smtlib" "z3"

#CHECK: {{[[:space:]]|^}}unsat{{[[:space:]]|$}}
run_equivalence_test "Dot64.mlir", "smtlib" "bitwuzla"

#CHECK: {{[[:space:]]|^}}unsatisfiable.{{[[:space:]]|$}}
run_equivalence_test "Dot64.mlir", "btor2" "bitwuzla"

#CHECK: {{[[:space:]]|^}}UNSATISFIABLE{{[[:space:]]|$}}
run_equivalence_test "Dot64.mlir", "aiger" "kissat"



echo "====================================================================== Unequal ======================================================================"


#CHECK: {{[[:space:]]|^}}sat{{[[:space:]]|$}}
run_equivalence_test "Dot64_truncation.mlir", "smtlib" "z3"

#CHECK: {{[[:space:]]|^}}sat{{[[:space:]]|$}}
run_equivalence_test "Dot64_truncation.mlir", "smtlib" "bitwuzla"

#CHECK: {{[[:space:]]|^}}satisfiable.{{[[:space:]]|$}}
run_equivalence_test "Dot64_truncation.mlir", "btor2" "bitwuzla"

#CHECK: {{[[:space:]]|^}}SATISFIABLE{{[[:space:]]|$}}
run_equivalence_test "Dot64_truncation.mlir", "aiger" "kissat"
