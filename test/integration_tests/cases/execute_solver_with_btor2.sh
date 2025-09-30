#!/usr/bin/bash

set -euo pipefail

# Execute MC
#   $1 - solver [btormc, bitwuzla ...]
#   $2 - input file
#   $3 - output directoy
solver="$1"
input_file="$2"
output_dir="$3"

# Convert to BTOR2
circt-opt --convert-hw-to-btor2  "$input_file" -o "$output_dir/top.log" &> "$output_dir/top.btor2"

# Run solver
# run_solver --solver "$solver" --inputfile "$output_dir/top.btor2"
equiv_fusion -p "run_solver --solver "$solver" --inputfile "$output_dir/top.btor2""



