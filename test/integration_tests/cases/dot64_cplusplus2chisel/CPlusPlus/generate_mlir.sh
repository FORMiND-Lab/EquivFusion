#!/usr/bin/bash

cgeist -S Dot64.cpp -o Dot64.mlir --remove-module-attributes --raise-scf-to-affine                            # generate Dot64.mlir

cgeist -S Dot64_truncation.cpp -o Dot64_truncation.mlir --remove-module-attributes --raise-scf-to-affine      # generate Dot64_truncation.mlir
