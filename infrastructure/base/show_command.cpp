//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/base/command.h"
#include "infrastructure/utils/log-util/log_util.h"


XUANSONG_NAMESPACE_HEADER_START

struct ShowCommand : public Command {
public:
    ShowCommand() : Command("show", "show executed information") {}
    ~ShowCommand() = default;

    ShowCommand(const ShowCommand &) = delete;
    ShowCommand &operator=(const ShowCommand &) = delete;

    void execute(const std::vector<std::string> &args) override {
        double totalTime = 0.0;
        const auto& executedCommandsInfo = getExecutedCommandsInfo();
        for (const auto &executedCommandInfo: executedCommandsInfo) {
            totalTime += executedCommandInfo.time;
        }

        log("\n");
        log("==================== Show executed commands information ====================\n");
        log("%20s    %-20s     %s\n", "No", "Command", "Time (seconds)");

        int idx = 1;
        for (const auto &executedCommandInfo: executedCommandsInfo) {
            double time = executedCommandInfo.time;
            log("%20d    %-20s    %8.6f %5.1f%%\n",
                idx++, executedCommandInfo.command->getName().c_str(), time, 100.0 * time / totalTime);
        }

        log("%20s    %-20s    %8.6f %5.1f%%\n", "Total", "", totalTime, 100.0 * totalTime / totalTime);
        log("============================================================================\n");
        log("\n");
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    %s \n", getName().c_str());
        log("   OPTIONS:\n");
        log("\n");
    }
} showCmd;

XUANSONG_NAMESPACE_HEADER_END
