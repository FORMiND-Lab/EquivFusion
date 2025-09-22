#!/usr/bin/bash

set -euo pipefail

# Execute LEC
#   $1 - solver [z3, bitwuzla ...]
#   $2 - first module name
#   $3 - second module name
#   $4 - output directory
#   $5 - input files
solver="$1"
name1="$2"
name2="$3"
out_dir="$4"
input_files="$5"

# EquivFusion: Construct Miter and Export SMT-LIB
EquivFusionLEC --c1 "$name1" --c2 "$name2" --emit-smtlib "$input_files" -o "$out_dir/output.smt"

# Run: Solver
equiv_fusion_solver --solver "$solver" --inputfile "$out_dir/output.smt"

