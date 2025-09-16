#include <iostream>
#include "infrastructure/log/log.h"
#include "solving/solver.hpp"

namespace XuanSong {
namespace Solver {

static int executeCommand(const std::string &command) {
    std::cout << "Command : " << command << std::endl;
    return system(command.c_str());
}

static int executeZ3(const std::string &command) {
    return executeCommand(command);
}

static int executeBitwuzla(const std::string &command) {
    return executeCommand(command);
}

static int executeBtorMC(const std::string &command) {
    std::string new_command = command + "--kind";
    return executeCommand(new_command);
}

static int executeKissat(const std::string &command) {
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

using SolverFunction = int (*)(const std::string &command);

static std::unordered_map<std::string, SolverFunction> kSolverToFunMap = {
    {"z3",          executeZ3},
    {"bitwuzla",    executeBitwuzla},
    {"btormc",      executeBtorMC},
    {"kissat",      executeKissat}
};

int executeSolver(const std::string &solverName, const std::string &inputFile, const std::string &options) {
    auto it = kSolverToFunMap.find(solverName);
    if (it == kSolverToFunMap.end()) {
        XuanSong::logError("Error Unsupported solver [%s]\n", solverName);
    }       
   
    std::string command = solverName + " " + inputFile + " " + options; 
    return it->second(command);
}

} // namespace Solver
} // namespace XuanSong
