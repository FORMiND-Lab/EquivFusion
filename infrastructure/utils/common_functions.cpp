#include <iostream>
#include <vector>
#include <unordered_map>
#include <readline/readline.h>
#include <readline/history.h>
#include "infrastructure/base/commandManager.h"
#include "infrastructure/log/log.h"
#include "infrastructure/utils/common_functions.h"

XUANSONG_NAMESPACE_HEADER_START

std::string nextToken(std::string &text, std::string sep) {
    size_t beginPos = text.find_first_not_of(sep);
    if (beginPos == std::string::npos) {
        return "";
    }

    size_t endPos = text.find_first_of(sep, beginPos);
    if (endPos == std::string::npos) {
        endPos = text.size();
    }

    std::string token = text.substr(beginPos, endPos - beginPos);
    text = text.substr(endPos);
    return token;
}

static char *readline_cmd_generator(const char *text, int state)
{
	static std::unordered_map<std::string, Command*>::iterator it;
	static int len;
    static std::unordered_map<std::string, Command*> command_register = CommandManager::getInstance()->getRegisteredCommands();


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

void runShell() { 
    // initilize the readline
    rl_readline_name = (char *)"EquivFusion";
    rl_attempted_completion_function = readline_completion;
    rl_basic_word_break_characters = (char *)" \t\n";

    CommandManager *commandMgr = CommandManager::getInstance();
    commandMgr->registerCommand();

    char *command = NULL;
    while ((command = readline("EquivFusion> ")) != NULL) {
        std::string commandStr(command);
        if (commandStr.find_first_not_of(" \t\n\r") == std::string::npos) { 
            free(command);
            command = NULL;
            continue;
        }

        add_history(command);
        commandStr.erase(0, commandStr.find_first_not_of(" \t\n\r"));

        if (commandStr.substr(0, 4) == "exit" || commandStr.substr(0, 4) == "quit") { 
            free(command);
            command = NULL;
            break;
        }
        
        std::string commandName = nextToken(commandStr, " \t\n\r");
        std::string token = nextToken(commandStr, " \t\n\r");
        std::vector<std::string> args;
        
        while (!token.empty()) {
            args.push_back(token);
            token = nextToken(commandStr, " \t\n\r");
        }

        commandMgr->executeCommand(commandName, args);

        free(command);
        command = NULL;
    }

    log("EquivFusion is shutting down...\n");
}

XUANSONG_NAMESPACE_HEADER_END

