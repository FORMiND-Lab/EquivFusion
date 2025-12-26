//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/base/command.h"
#include "infrastructure/base/commandManager.h"
#include "infrastructure/utils/log-util/log_util.h"

XUANSONG_NAMESPACE_HEADER_START

struct HelpCmd : public Command {
public: 
    HelpCmd() : Command("help", "display help information") {}
    ~HelpCmd() = default;

    HelpCmd(const HelpCmd &) = delete;
    HelpCmd &operator=(const HelpCmd &) = delete;

    void preExecute() override {
        
    }

    void execute(const std::vector<std::string> &args) override {
        CommandManager *commandMgr = CommandManager::getInstance();

        if (args.empty()) {
            std::map<std::string, Command *> registeredCommands = commandMgr->getRegisteredCommands();
            log("\n");
            for (auto &command : registeredCommands) {
                log("    %-30s %s\n", command.first.c_str(), command.second->getDescription().c_str());
            }
            log("\n");
        } else {
            std::string commandName = args[0];
            if (commandMgr->hasCommand(commandName)) { 
                Command *command = commandMgr->getCommand(commandName);
                command->help();
            } else {
                log("Error: Command '%s' not found\n", commandName.c_str());
            }
        }
    }

    void postExecute() override {
        
    }

    void help() override {
        log("\n");
        log("    help ------------------- list all commands\n");
        log("    help <command> --------- print help information for given command\n");
        log("\n");
    }


} helpCmd;

XUANSONG_NAMESPACE_HEADER_END


