
```
cases/
├── addition_swap_patameters/
│   └── addition_swap_patameters.sh: Executable test script 
│   └── input.mlir: input file 
├── ...
├── execute_solver_with_cnf.sh:    common shell [mlir => aiger => miter to cnf => run solver (kissat, ...) ]
├── execute_solver_with_smt2.sh:   common shell [mlir => smt => run solver (z3, bitwuzla, ...) ]
└── execute_solver_with_btor2.sh:  common shell [mlir => btor2 => run solver (btormc, bitwuzla, ...) ]
```
