#include "infrastructure/base/commandManager.h"

#include "libs/Backend/aiger/aiger.h"
#include "libs/Backend/btor2/btor2.h"
#include "libs/Backend/mlir/mlir.h"
#include "libs/Backend/smt/smt.h"

XUANSONG_NAMESPACE_HEADER_START

template<typename Impl>
struct BackendCommand : public Command {
public:
    BackendCommand(const std::string& name, const std::string& description) : Command(name, description) {}
    ~BackendCommand() override = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        mlir::MLIRContext& context = *CommandManager::getInstance()->getGlobalContext();
        mlir::ModuleOp inputModule = CommandManager::getInstance()->getModuleOp();
        mlir::OwningOpRef<mlir::ModuleOp> outputModule;
        if (Impl::run(args, context, inputModule, outputModule)) {
            CommandManager::getInstance()->setModuleOp(outputModule);
        }
    }

    void postExecute() override {
    }

    void help() override {
        Impl::help(getName(), getDescription());
    }
};

BackendCommand<WriteSMTImpl>    writeSMTCmd("write_smt", "write smt");
BackendCommand<WriteAIGERImpl>  writeAIGERCmd("write_aiger", "write aiger");
BackendCommand<WriteBTOR2Impl>  writeBTOR2Cmd("write_btor2", "write btor2");
BackendCommand<WriteMLIRImpl>   writeMLIRCmd("write_mlir",  "write mlir");

XUANSONG_NAMESPACE_HEADER_END


