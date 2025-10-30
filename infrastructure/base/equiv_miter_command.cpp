#include "infrastructure/base/equivfusionManager.h"
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
        mlir::MLIRContext& context = *EquivFusionManager::getInstance()->getGlobalContext();
        mlir::ModuleOp inputModule = EquivFusionManager::getInstance()->getModuleOp();
        mlir::OwningOpRef<mlir::ModuleOp> outputModule;
        if (EquivMiterTool::run(args, context, inputModule, outputModule)) {
            EquivFusionManager::getInstance()->setModuleOp(outputModule);
        }
    }

    void postExecute() override {
    }

    void help() override {
        EquivMiterTool::help(getName(), getDescription());
    }
} equivMiterCmd;

XUANSONG_NAMESPACE_HEADER_END


