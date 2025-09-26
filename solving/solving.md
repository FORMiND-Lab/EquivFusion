for solving.


# Install Solver
see CMakeLists.txt

## Install
```
cd EquivFusion
mkdir build
cd build
cmake .. -G Ninja
ninja install_solvers

```
## Target
EquivFusion/solving/dep_solvers/


# Run Solver
```
RunSolver::runSolver(solverName, inputFile, options)
```

## Files
```
├── run_solver.cpp/run_solver.h   [runSolver] class for run solvers
├── solver.cpp/solver.h:          [Solver]: abstract class for solvers; Specific solvers inherit from this class
└── solver_definitions.sh         [SOLVER_LIST]
```

# Supported Solver [see SOLVER_LIST]
- z3
- bitwuzla
- btormc
- kissat

# Extend Solver
- Install solver
  - add config in scripts/solvers/config_xxx.sh
  - add xxx in solving/CMakeLists.sh `SUPPORTED_SOLVERS`
- Run Solver Code
  - add solver into SOLVER_LIST [solver_definitions.sh]
  - add implementation of xxxSolver:run() in solver.cpp

