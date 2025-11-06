#include <iostream>
#include "libs/cxxopts/cxxopts.hpp"
#include "infrastructure/utils/log/log.h"
#include "infrastructure/base/commandManager.h"
#include "infrastructure/kernel/run_command.h"

int main(int argc, char** argv) { 
    bool run_shell = true;
    std::vector<std::string> commands;

    cxxopts::Options options(argv[0], "EquivFusion -- Silent as Pine, Precise as Logic");
    options.add_options("operation")
        ("p,command", "execute <command>)", cxxopts::value<std::vector<std::string>>(), "<commands>")
        ("H", "print the command list")
        ("h,help", "print this help message.")
    ;

    XuanSong::CommandManager *commandMgr = XuanSong::CommandManager::getInstance();
    commandMgr->registerCommand();

    try {
        auto result = options.parse(argc, argv);

        if (result.count("H")) { 
            XuanSong::runCommand("help");
            run_shell = false;
        }

        if (result.count("h")) { 
            XuanSong::log("%s\n", options.help().c_str());
            run_shell = false;
        }

        if (result.count("p")) {
            auto cmds = result["p"].as<std::vector<std::string>>();
            commands.insert(commands.end(), cmds.begin(), cmds.end());
            run_shell = false;
        }
    } catch (const cxxopts::exceptions::parsing& e) {
        XuanSong::log("Error parsing options: %s\n", e.what());
        XuanSong::log("Run '%s --help' for help.\n", argv[0]);
        exit(1);
	}

    for (auto it = commands.begin(); it != commands.end(); it++) {
        XuanSong::runCommand(*it);
    }

    if (run_shell) { 
        XuanSong::runShell();
    }
    
    return 0;
}

