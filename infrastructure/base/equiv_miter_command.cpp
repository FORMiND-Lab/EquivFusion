#include "infrastructure/base/command.h"
#include "infrastructure/utils/log-util/log_util.h"

#include "libs/Tools/EquivMiter/equiv_miter.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterCommand : public Command {
public:
    EquivMiterCommand() : Command("equiv_miter", "construct miter for logical equivalence check") {}
    ~EquivMiterCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        if (!EquivMiterTool::executeMiter(args)) {
            logError("Command 'equiv_miter' failed!\n");
        }
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());;
        log("   USAGE:    %s <--specModule name> <--implModule name> [options]\n", getName().c_str());
        log("   OPTIONS:\n");
        log("       --print-ir ----------------------------- Print IR after pass\n");
        log("       --specModule <module name> ------------- Specify a named module for the specification circuit\n");
        log("       --implModule <module name> ------------- Specify a named module for the implementation circuit\n");
        log("       --mitermode ---------------------------- MiterMode [smtlib, aiger, btor2], default is smtlib\n");
        log("       -o ------------------------------------- Output filename\n");
        log("\n\n");
    }
} equivMiterCmd;

XUANSONG_NAMESPACE_HEADER_END


