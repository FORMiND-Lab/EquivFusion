#include <iostream>
#include "solving/solver-runner/solver_declare.h"

namespace XuanSong {

int Z3Solver::run(const std::string &command) {
    return executeCommand(command);
}

int BitwuzlaSolver::run(const std::string &command) {
    return executeCommand(command);
}

int BtorMCSolver::run(const std::string &command) {
    std::string actual_command = command + " --kind";
    return executeCommand(actual_command);
}

int KissatSolver::run(const std::string &command) {
    int status = executeCommand(command);
    if (status == -1) {
        return -1;
    }

    if (WIFEXITED(status)) {
        int result = WEXITSTATUS(status);
        if (result == 20 || result == 10) {
            // unsat or sat
            return 0;
        }
    }

    return -1;
}

} // namespace XuanSong
