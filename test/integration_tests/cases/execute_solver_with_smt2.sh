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
input_files="${@:5}"

# Construct Miter and Export SMT-LIB
equiv_fusion -p "equiv_miter --c1 "$name1" --c2 "$name2" $input_files --mitermode smtlib -o "$out_dir/miter.smt""

# Solver runner
equiv_fusion -p "solver_runner --solver "$solver" --inputfile "$out_dir/miter.smt""

