#ifndef SOLVER_RUNNER_H
#define SOLVER_RUNNER_H

#include <unordered_map>
#include <memory>

#include "solving/solver-runner/solver_declare.h"

namespace XuanSong {

class SolverRunner {
private:
    static std::unordered_map<std::string, std::unique_ptr<Solver>> solvers;
    static void initializeSolvers();

    static Solver* getSolver(const std::string &solverName) {
        auto it = solvers.find(solverName);
        return it != solvers.end() ? it->second.get() : nullptr;
    }

    static int run(Solver* solver, const std::string &command);
public:
    static int run(const std::string &solverName, const std::string &inputFile, const std::string& options);
};

} // namespace XuanSong

#endif //SOLVER_RUNNER_H


