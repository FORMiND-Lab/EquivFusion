add_subdirectory(solver-runner)

# -------------------------------------------
# Install Solvers
# -------------------------------------------
set(SOLVERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dep_solvers)
set(SCRIPT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/solvers/install_solvers.sh)

# macro: add single solver target
macro(create_solver_target SOLVER_NAME)
    message(STATUS "Add custom target install_${SOLVER_NAME}.")
    add_custom_target(install_${SOLVER_NAME}
        COMMAND SOLVERS_DIR=${SOLVERS_DIR} INSTALL_DIR=${OUTPUT_BIN_DIR} bash ${SCRIPT_FILE} --${SOLVER_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Installing ${SOLVER_NAME}..."
    )
    list(APPEND SOLVER_TARGETS install_${SOLVER_NAME})
endmacro()

# Config Binaries for each solver
set(SUPPORTED_SOLVERS aiger kissat boolector bitwuzla z3)

# Add single solver target
set(SOLVER_TARGETS "")
foreach(SOLVER IN LISTS SUPPORTED_SOLVERS)
    create_solver_target("${SOLVER}")
endforeach()

add_custom_target(install_solvers
    COMMENT "Installing all solvers ..."
    DEPENDS ${SOLVER_TARGETS}
)

