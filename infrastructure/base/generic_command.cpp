#include "infrastructure/base/commandManager.h"

#include "libs/Frontend/read_mlir.h"

#include "libs/Backend/write_smt.h"
#include "libs/Backend/write_aiger.h"
#include "libs/Backend/write_btor2.h"

#include "libs/Tools/EquivMiter/equiv_miter.h"

XUANSONG_NAMESPACE_HEADER_START

template<typename Impl>
struct GenericCommand : public Command {
public:
    GenericCommand(const std::string& name, const std::string& description) : Command(name, description) {}
    ~GenericCommand() override = default;

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

GenericCommand<ReadMLIRImpl>   readMLIRCmd("read_mlir", "read mlir");

GenericCommand<WriteSMTImpl>   writeSMTCmd("write_smt", "write smt");
GenericCommand<WriteAIGERImpl> writeAIGERCmd("write_aiger", "write aiger");
GenericCommand<WriteBTOR2Impl> writeBTOR2Cmd("write_btor2", "write btor2");

GenericCommand<EquivMiterImpl> equivMiterCmd("equiv_miter", "construct miter for logical equivalence check");


XUANSONG_NAMESPACE_HEADER_END


