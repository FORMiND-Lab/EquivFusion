//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SOLVER_DECLARE_HPP
#define SOLVER_DECLARE_HPP

#include <string>

#define SOLVER_LIST \
    SOLVER(Z3Solver, "z3") \
    SOLVER(BitwuzlaSolver, "bitwuzla") \
    SOLVER(BtorMCSolver, "btormc")    \
    SOLVER(KissatSolver, "kissat")    \

namespace XuanSong {

class Solver {
public:
    virtual ~Solver() = default;
    virtual int run(const std::string &inputFile, const std::string &options) = 0;
    virtual std::string getName() const = 0;

protected:
    int executeCommand(const std::string &command) {
        return system(command.c_str());
    }

    /// Convert AIG to CNF
    int convertToCNF(const std::string &aigFile, const std::string &cnfFile) {
        std::string command = "aigtocnf " + aigFile + " " + cnfFile;
        return executeCommand(command);
    }
};

// Declare Solver Class
#define SOLVER(ClassName, SolverName) \
class ClassName : public Solver { \
public: \
    std::string getName() const override { return SolverName; } \
    int run(const std::string &inputFile, const std::string &options) override; \
};

SOLVER_LIST
#undef SOLVER


} // namespace XuanSong

#endif //SOLVER_DECLARE_HPP


