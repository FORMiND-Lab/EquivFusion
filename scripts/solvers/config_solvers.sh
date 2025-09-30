#!/bin/bash

# Solvers descriptions
declare -A SOLVERS_DESC=(
    [kissat]="Install kissat (SAT solver)"
    [aiger]="Install aiger (AIGER format tools)"
    [boolector]="Install boolector (SMT solver for bit-vectors)"
    [bitwuzla]="Install bitwuzla (SMT solver)"
)

# Supported solvers
SUPPORTED_SOLVERS=("${!SOLVERS_DESC[@]}")

# Show help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    for solver in "${SUPPORTED_SOLVERS[@]}"; do
        printf "  --%-12s: %s\n" "$solver" "${SOLVERS_DESC[$solver]}"
    done
    echo "  -h, --help    : Show help information"
}

# Check solver is supported or not
is_supported_solver() {
    local solver_name="$1"
    [[ " ${SUPPORTED_SOLVERS[*]} " =~ " ${solver_name} " ]]
}
