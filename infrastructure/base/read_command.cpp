#include "infrastructure/base/equivfusionManager.h"
#include "infrastructure/base/command.h"

#include "libs/Read/read_mlir.h"

#include "libs/Tools/EquivMiter/equiv_miter.h"

XUANSONG_NAMESPACE_HEADER_START

template<typename Impl>
struct ReadCommand : public Command {
public:
    ReadCommand(const std::string& name, const std::string& description) : Command(name, description) {}
    ~ReadCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        mlir::MLIRContext& context = *EquivFusionManager::getInstance()->getGlobalContext();
        mlir::ModuleOp inputModule = EquivFusionManager::getInstance()->getModuleOp();
        mlir::OwningOpRef<mlir::ModuleOp> outputModule;
        if (Impl::run(args, context, inputModule, outputModule)) {
            EquivFusionManager::getInstance()->setModuleOp(outputModule);
        }
    }

    void postExecute() override {
    }

    void help() override {
        Impl::help(getName(), getDescription());
    }
};

ReadCommand<ReadMLIRImpl>    readMLIRCmd("read_mlir", "read mlir");

ReadCommand<EquivMiterImpl>  equivMiterCmd("equiv_miter", "construct miter for logical equivalence check");


XUANSONG_NAMESPACE_HEADER_END


