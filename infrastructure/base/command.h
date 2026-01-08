//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <chrono>
#include <string>
#include <vector>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START

struct ExecutedCommandInfo {
    struct Command *command;
    double time;  // seconds

    ExecutedCommandInfo(Command *cmd,
                        const std::chrono::steady_clock::time_point &startTime,
                        const std::chrono::steady_clock::time_point &endTime) : command(cmd) {
        std::chrono::duration<double> duration_s(endTime - startTime);
        time = duration_s.count();
    }
};

struct Command {
private: 
    std::string name_;
    std::string description_;
    Command *nextCommand_;
    std::chrono::steady_clock::time_point startTime_;

public:
    Command(const std::string &name, const std::string &description);
    virtual ~Command() = default;

    Command(const Command &) = delete;
    Command &operator=(const Command &) = delete;

    virtual void execute(const std::vector<std::string> &args) = 0;
    virtual void help() = 0;

    std::string getName() const;
    std::string getDescription() const;
    Command *getNextCommand() const;

    void prevExecute();
    void postExecute();

private:
    static std::vector<ExecutedCommandInfo> executedCommandsInfo_;
protected:
    static std::vector<ExecutedCommandInfo>& getExecutedCommandsInfo();
};

extern Command *firstCommand;


XUANSONG_NAMESPACE_HEADER_END

