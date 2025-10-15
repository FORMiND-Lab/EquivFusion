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

# Supported Solver [see SOLVER_LIST]
- z3
- bitwuzla
- btormc
- kissat

# Extend Solver
- Install solver
  - add config in scripts/solvers/config_xxx.sh
  - add xxx in solving/CMakeLists.txt `SUPPORTED_SOLVERS`
- SolverRunner Code
  - add solver into SOLVER_LIST [solver-runner/solver_declare.h]
  - add implementation of xxxSolver:run() [solver-runner/solver_declare.cpp]

