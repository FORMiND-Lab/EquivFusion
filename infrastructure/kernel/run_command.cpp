//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>
#include <readline/readline.h>
#include <readline/history.h>
#include "infrastructure/base/commandManager.h"
#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/utils/string-util/string_util.h"
#include "infrastructure/kernel/run_command.h"

XUANSONG_NAMESPACE_HEADER_START

static char *readline_cmd_generator(const char *text, int state)
{
	static std::map<std::string, Command*>::iterator it;
	static int len;
    static std::map<std::string, Command*> command_register = CommandManager::getInstance()->getRegisteredCommands();


	if (!state) {
		it = command_register.begin();
		len = strlen(text);
	}

	for (; it != command_register.end(); it++) {
		if (it->first.compare(0, len, text) == 0)
			return strdup((it++)->first.c_str());
	}
	return NULL;
}

static char **readline_completion (const char *text, int start, int) {
    if (start == 0) { 
        return rl_completion_matches(text, readline_cmd_generator);
    }

    return NULL;
}

void runCommand(std::string commandStr) {  
    add_history(commandStr.c_str());

    std::string commandName = Utils::StringUtil::nextToken(commandStr, " \t\n\r");
    std::string token = Utils::StringUtil::nextToken(commandStr, " \t\n\r");
    std::vector<std::string> args;
    
    while (!token.empty()) {
        args.push_back(token);
        token = Utils::StringUtil::nextToken(commandStr, " \t\n\r");
    }

    CommandManager *commandMgr = CommandManager::getInstance();
    commandMgr->executeCommand(commandName, args);
}

void runShell() { 
    XuanSong::logEquivFusionBanner();

    // initilize the readline
    rl_readline_name = (char *)"EquivFusion";
    rl_attempted_completion_function = readline_completion;
    rl_basic_word_break_characters = (char *)" \t\n";

    char *command = NULL;
    while ((command = readline("EquivFusion> ")) != NULL) {
        std::string commandStr(command);
        if (commandStr.find_first_not_of(" \t\n\r") == std::string::npos) { 
            free(command);
            command = NULL;
            continue;
        }

        commandStr.erase(0, commandStr.find_first_not_of(" \t\n\r"));

        if (commandStr.substr(0, 4) == "exit" || commandStr.substr(0, 4) == "quit") { 
            free(command);
            command = NULL;
            break;
        }

        runCommand(commandStr);

        free(command);
        command = NULL;
    }

    log("EquivFusion is shutting down...\n");
}

XUANSONG_NAMESPACE_HEADER_END

