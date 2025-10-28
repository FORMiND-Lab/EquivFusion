#include "infrastructure/base/commandManager.h"
#include "infrastructure/log/log.h"

#include "backend/write_smt.h"
#include "backend/write_aiger.h"
#include "backend/write_btor2.h"

XUANSONG_NAMESPACE_HEADER_START

template<typename BackendImpl>
struct BackendCommand : public Command {
public:
    BackendCommand(const std::string &name, const std::string& description) : Command(name, description) {}
    ~BackendCommand() override = default;

    void preExecute() override {}
    
    void execute(const std::vector<std::string> &args) override {
        BackendImpl impl;
        
        if (!impl.initOptions(args)) {
            log("[%s]: options error\n\n", getName().c_str());
            return;
        }
        
        mlir::MLIRContext& context = *CommandManager::getInstance()->getGlobalContext();
        auto inputModuleOp = CommandManager::getInstance()->getModuleOp();
        if (!impl.run(context, inputModuleOp)) {
            log("[%s]: execute error\n\n", getName().c_str());
        }
    }
    
    void postExecute() override {}
    
    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    %s [-o filename]\n", getName().c_str());
        log("   OPTIONS:\n");
        log("       -o <filename>           - Output filename\n");
        log("   Example:");
        log("       %s -o test.output", getName().c_str());
        log("\n\n");
    }
};


BackendCommand<WriteSMTImpl>   writeSMTCmd("write_smt", "write smt");
BackendCommand<WriteAIGERImpl> writeAIGERCmd("write_aiger", "write aiger");
BackendCommand<WriteBTOR2Impl> writeBTOR2Cmd("write_btor2", "write btor2");

XUANSONG_NAMESPACE_HEADER_END

