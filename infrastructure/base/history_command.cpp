#include <cstdio>
#include <readline/history.h>
#include <readline/readline.h>
#include "infrastructure/base/command.h"
#include "infrastructure/log/log.h"


XUANSONG_NAMESPACE_HEADER_START

struct HistoryCmd : public Command {
public: 
    HistoryCmd() : Command("history", "show last interactive commands") {}
    ~HistoryCmd() override = default;

    HistoryCmd(const HistoryCmd &) = delete;
    HistoryCmd &operator=(const HistoryCmd &) = delete;

    void preExecute() override {

    }

    void execute(const std::vector<std::string> &args) override {
        for(HIST_ENTRY **list = history_list(); *list != NULL; list++) {
            log("%s\n", (*list)->line);
        }

        log("\n");
    }

    void postExecute() override {
        

    }

    void help() override { 
        log("\n");
        log("    history ---------------- show last interactive commands\n");
        log("\n");
    }
} historyCmd;

XUANSONG_NAMESPACE_HEADER_END
