#!/usr/bin/bash
set -euo pipefail

# Execute LEC
#   $1 - solver [btormc, bitwuzla ...]
#   $2 - first module name
#   $3 - second module name
#   $4 - output directory
#   $5 - input files
solver="$1"
name1="$2"
name2="$3"
out_dir="$4"
input_files="${@:5}"

echo $input_files
# Construct Miter and Export to AIGER
equiv_fusion -p "read_mlir $input_files" \
             -p "equiv_miter --c1 "$name1" --c2 "$name2" --mitermode aiger" \
             -p "write_aiger "$out_dir/miter.aiger""

# Convert aiger to cnf
aigtocnf "$out_dir/miter.aiger" "$out_dir/miter.cnf"

# Solver runner
equiv_fusion -p "solver_runner --solver "$solver" --inputfile "$out_dir/miter.cnf""

