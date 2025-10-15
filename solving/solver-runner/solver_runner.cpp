#include <iostream>
#include <cassert>

#include "solving/solver-runner/solver_runner.h"

namespace XuanSong {

std::unordered_map<std::string, std::unique_ptr<Solver>> SolverRunner::solvers;

void SolverRunner::initializeSolvers() {
    if (solvers.empty()) {
    #define SOLVER(ClassName, SolverName) \
        solvers[SolverName] = std::make_unique<ClassName>();

    SOLVER_LIST
    #undef SOLVER
    }
}

int SolverRunner::run(Solver* solver, const std::string &command) {
    assert(solver);
    return solver->run(command);
}

int SolverRunner::run(const std::string &solverName, const std::string &inputFile, const std::string& options) {
    initializeSolvers();

    auto solver = getSolver(solverName);
    if (!solver) {
        std::cerr << "Unsupported Solver " << solverName << std::endl;
        return -1;
    }
    assert(solverName == solver->getName());

    std::string command = solverName + " " + options + " " + inputFile;
    return run(solver, command);
}

} // namespace XuanSong
