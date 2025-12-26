//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

public:
    static int run(const std::string &solverName, const std::string &inputFile, const std::string& options);
};

} // namespace XuanSong

#endif //SOLVER_RUNNER_H


