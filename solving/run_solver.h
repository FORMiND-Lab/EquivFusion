#ifndef RUN_SOLVER_H
#define RUN_SOLVER_H

#include <unordered_map>
#include <memory>

#include "solving/solver.h"

namespace XuanSong {

class RunSolver {
private:
    static std::unordered_map<std::string, std::unique_ptr<Solver>> solvers;
    static void initializeSolvers();

    static Solver* getSolver(const std::string &solverName) {
        auto it = solvers.find(solverName);
        return it != solvers.end() ? it->second.get() : nullptr;
    }

    static int runSolver(Solver* solver, const std::string &command);
public:
    static int runSolver(const std::string &solverName, const std::string &inputFile, const std::string& options);
};

} // namespace XuanSong

#endif //RUN_SOLVER_H


