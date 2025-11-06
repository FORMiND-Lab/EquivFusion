#include "infrastructure/base/commandManager.h"
#include "infrastructure/utils/log/log.h"


XUANSONG_NAMESPACE_HEADER_START

CommandManager *CommandManager::instance_ = nullptr;

CommandManager *CommandManager::getInstance() {
    if (instance_ == nullptr) {
        instance_ = new CommandManager();
    }
    return instance_;
}

void CommandManager::registerCommand() {
    Command *cmd = firstCommand;
    while (cmd != nullptr) {
        if (hasCommand(cmd->getName())) { 
            logError("Unable to register command %s, it already exists", cmd->getName().c_str());
        }

        registeredCommands_[cmd->getName()] = cmd;
        cmd = cmd->getNextCommand();
    }
}

bool CommandManager::hasCommand(const std::string &name) const {
    return registeredCommands_.find(name) != registeredCommands_.end();
}

Command *CommandManager::getCommand(const std::string &name) const {
    auto it = registeredCommands_.find(name);
    if (it == registeredCommands_.end()) {
        return nullptr;
    }
    return it->second;
}

std::map<std::string, Command *> CommandManager::getRegisteredCommands() const {
    return registeredCommands_;
}

void CommandManager::executeCommand(const std::string &name, const std::vector<std::string> &args) { 
    if (!hasCommand(name)) { 
        log("Error: Command '%s' not found\n", name.c_str());
        return;
    }

    Command *cmd = getCommand(name);
    cmd->execute(args);
}

XUANSONG_NAMESPACE_HEADER_END
