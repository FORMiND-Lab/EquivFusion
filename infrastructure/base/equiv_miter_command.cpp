#include "infrastructure/base/command.h"
#include "libs/Tools/EquivMiter/equiv_miter.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterCommand : public Command {
public:
    EquivMiterCommand() : Command("equiv_miter", "construct miter for logical equivalence check") {}
    ~EquivMiterCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        EquivMiterTool::execute(args);
    }

    void postExecute() override {
    }

    void help() override {
        EquivMiterTool::help(getName(), getDescription());
    }
} equivMiterCmd;

XUANSONG_NAMESPACE_HEADER_END


