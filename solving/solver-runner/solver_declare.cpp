#include "solving/solver-runner/solver_declare.h"

namespace XuanSong {

int Z3Solver::run(const std::string &inputFile, const std::string &options) {
    std::string command = getName() + " " + inputFile + " " + options;
    return executeCommand(command);
}

int BitwuzlaSolver::run(const std::string &inputFile, const std::string &options) {
    std::string command = getName() + " " + inputFile + " " + options;
    return executeCommand(command);
}

int BtorMCSolver::run(const std::string &inputFile, const std::string &options) {
    std::string command = getName() + " " + inputFile + " " + options + " --kind";
    return executeCommand(command);
}

int KissatSolver::run(const std::string &inputFile, const std::string &options) {
    std::string cnfFile = inputFile + ".cnf";
    if (convertToCNF(inputFile, cnfFile) != 0) {
        return -1;
    }

    std::string command = getName() + " " + cnfFile + " " + options;
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
