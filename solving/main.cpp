#include <iostream>
#include <cassert>
#include "libs/cxxopts/cxxopts.hpp"
#include "infrastructure/log/log.h"

#include "solving/solver.hpp"

int main(int argc, char **argv) {
    std::string solver = "";
    std::string inputFile = "";
    std::string opts = "";

    cxxopts::Options options(argv[0], "Solver");
    options.add_options("operation")
        ("H", "print the command list")
        ("h,help", "print this help message.")
        ("solver", "use <solver> for select solver", cxxopts::value<std::string>(), "<solver>")
        ("inputfile","use <inputfile> for solver", cxxopts::value<std::string>(), "<inputfile>")
        ("opts", "use <opts> for solver", cxxopts::value<std::string>(), "<opts>")
    ;

    try {
        auto result = options.parse(argc, argv);
        
        if (result.count("h")) {
            XuanSong::log("%s\n", options.help().c_str());
            return 0;
        }

        if (result.count("solver"))     solver = result["solver"].as<std::string>();
        if (result.count("inputfile"))  inputFile = result["inputfile"].as<std::string>();
        if (result.count("opts"))       opts = result["opts"].as<std::string>();
    } catch (const cxxopts::exceptions::parsing& e) {
        XuanSong::log("Error parsing options: %s\n", e.what());
        XuanSong::log("Run '%s --help' for help.\n", argv[0]);
        return -1;
    }

    int result = XuanSong::Solver::executeSolver(solver, inputFile, opts);
    return result;
}
