#!/usr/bin/bash

set -euo pipefail

# Execute LEC
#   $1 - solver [kissat]
#   $2 - first file path  (.mlir)
#   $3 - second file path (.mlir)
#   $4 - output directory
#   $5 - miter tool [abc, aiger]
solver="$1"
file1="$2"
file2="$3"
out_dir="$4"
miter_tool="$5"

name1=$(basename "$file1" .mlir)
name2=$(basename "$file2" .mlir)

# CIRCT: COMB Dialect Convert to AIG Dialect
circt-opt --convert-comb-to-aig "$file1" -o "$out_dir/${name1}_aig.mlir"
circt-opt --convert-comb-to-aig "$file2" -o "$out_dir/${name2}_aig.mlir"

# CIRCT: Export AIG
circt-translate --export-aiger "$out_dir/${name1}_aig.mlir" -o "$out_dir/${name1}.aig"
circt-translate --export-aiger "$out_dir/${name2}_aig.mlir" -o "$out_dir/${name2}.aig"

# Miter: [abc, aiger]
{
  case "$miter_tool" in
    "abc")
      # abc: miter to cnf
      abc -c "miter \"$out_dir/${name1}.aig\" \"$out_dir/${name2}.aig\"; write_cnf \"$out_dir/miter.cnf\""
      ;;
    "aiger")
      # aiger: miter, convert aig to cnf
      aigmiter "$out_dir/${name1}.aig" "$out_dir/${name2}.aig" -o "$out_dir/miter.aig"
      aigtocnf "$out_dir/miter.aig" "$out_dir/miter.cnf"
      ;;
    *)
      echo "Error: Unsupported miter tool '$miter_tool'. Supported tools [abc, aiger]"
      exit 1
      ;;
  esac
}

# Run solver
# run_solver --solver "$solver" --inputfile "$out_dir/miter.cnf"
equiv_fusion -p "run_solver --solver "$solver" --inputfile "$out_dir/miter.cnf""

