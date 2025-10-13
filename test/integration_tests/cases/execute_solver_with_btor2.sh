#!/usr/bin/bash

set -euo pipefail

# Execute MC
#   $1 - solver [btormc, bitwuzla ...]
#   $2 - first module name
#   $3 - second module name
#   $4 - output directory
#   $5 - input files
solver="$1"
name1="$2"
name2="$3"
out_dir="$4"
input_files="$5"

# Construct Miter and Export to BTOR2
equiv_miter --c1 "$name1" --c2 "$name2" "$input_files" --btor2 -o "$out_dir/miter.btor2"

# Run solver
## run_solver --solver "$solver" --inputfile "$output_dir/top.btor2"
equiv_fusion -p "run_solver --solver "$solver" --inputfile "$out_dir/miter.btor2""



