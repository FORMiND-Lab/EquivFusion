#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/base/command.h"
#include "infrastructure/utils/log/log.h"

#include "libs/Tools/EquivMiter/equiv_miter.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterCommand : public Command {
public:
    EquivMiterCommand() : Command("equiv_miter", "construct miter for logical equivalence check") {}
    ~EquivMiterCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        if (!EquivMiterTool::run(args)) {
            logError("Command 'equiv_miter' failed!\n");
        }
    }

    void postExecute() override {
    }

    void help() override {
        EquivMiterTool::help(getName(), getDescription());
    }
} equivMiterCmd;

XUANSONG_NAMESPACE_HEADER_END


