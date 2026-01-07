#!/usr/bin/bash

cgeist -S Dot64.c -o Dot64.mlir --remove-module-attributes --raise-scf-to-affine      # generate Dot64.mlir
sed -i 's/memref<?xi16>/memref<64xi16>/g' Dot64.mlir


cgeist -S Dot64_Unequal.c -o Dot64_Unequal.mlir --remove-module-attributes --raise-scf-to-affine      # generate Dot64_Unequal.mlir
sed -i 's/memref<?xi16>/memref<64xi16>/g' Dot64_Unequal.mlir