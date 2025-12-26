//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#pragma once

#include <map>
#include "infrastructure/base/command.h"

XUANSONG_NAMESPACE_HEADER_START

struct CommandManager {
private: 
    static CommandManager *instance_;
    std::map<std::string, Command *> registeredCommands_;

public: 
    CommandManager() = default;
    ~CommandManager() = default;

    CommandManager(const CommandManager &) = delete;
    CommandManager &operator=(const CommandManager &) = delete;

    static CommandManager *getInstance();

    void registerCommand();

    bool hasCommand(const std::string &name) const;

    Command *getCommand(const std::string &name) const;

    std::map<std::string, Command *> getRegisteredCommands() const;

    void executeCommand(const std::string &name, const std::vector<std::string> &args);
};

XUANSONG_NAMESPACE_HEADER_END
