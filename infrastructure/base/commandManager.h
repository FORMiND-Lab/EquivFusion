#pragma once

#include <unordered_map>
#include "infrastructure/base/command.h"

XUANSONG_NAMESPACE_HEADER_START

struct CommandManager {
private: 
    static CommandManager *instance_;
    std::unordered_map<std::string, Command *> registeredCommands_;

public: 
    CommandManager() = default;
    ~CommandManager() = default;

    CommandManager(const CommandManager &) = delete;
    CommandManager &operator=(const CommandManager &) = delete;

    static CommandManager *getInstance();

    void registerCommand();

    bool hasCommand(const std::string &name) const;

    Command *getCommand(const std::string &name) const;

    std::unordered_map<std::string, Command *> getRegisteredCommands() const;

    void executeCommand(const std::string &name, const std::vector<std::string> &args);
};

XUANSONG_NAMESPACE_HEADER_END
