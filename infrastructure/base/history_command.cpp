//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <readline/history.h>
#include <readline/readline.h>
#include "infrastructure/base/command.h"
#include "infrastructure/utils/log-util/log_util.h"


XUANSONG_NAMESPACE_HEADER_START

struct HistoryCmd : public Command {
public: 
    HistoryCmd() : Command("history", "show last interactive commands") {}
    ~HistoryCmd() = default;

    HistoryCmd(const HistoryCmd &) = delete;
    HistoryCmd &operator=(const HistoryCmd &) = delete;

    void execute(const std::vector<std::string> &args) override {
        for(HIST_ENTRY **list = history_list(); *list != NULL; list++) {
            log("%s\n", (*list)->line);
        }

        log("\n");
    }

    void help() override { 
        log("\n");
        log("    history ---------------- show last interactive commands\n");
        log("\n");
    }
} historyCmd;

XUANSONG_NAMESPACE_HEADER_END
