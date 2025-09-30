#include <iostream>
#include <cassert>

#include "solving/run_solver.h"

namespace XuanSong {

std::unordered_map<std::string, std::unique_ptr<Solver>> RunSolver::solvers;

void RunSolver::initializeSolvers() {
    if (solvers.empty()) {
    #define SOLVER(ClassName, SolverName) \
        solvers[SolverName] = std::make_unique<ClassName>();

    SOLVER_LIST
    #undef SOLVER
    }
}

int RunSolver::runSolver(Solver* solver, const std::string &command) {
    assert(solver);
    return solver->run(command);
}

int RunSolver::runSolver(const std::string &solverName, const std::string &inputFile, const std::string& options) {
    initializeSolvers();

    auto solver = getSolver(solverName);
    if (!solver) {
        std::cerr << "Unsupported Solver " << solverName << std::endl;
        return -1;
    }
    assert(solverName == solver->getName());

    std::string command = solverName + " " + options + " " + inputFile;
    return runSolver(solver, command);
}

} // namespace XuanSong
