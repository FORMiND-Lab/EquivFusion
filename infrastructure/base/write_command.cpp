#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/base/command.h"
#include "libs/Write/aiger/aiger.h"
#include "libs/Write/btor2/btor2.h"
#include "libs/Write/smt/smt.h"

XUANSONG_NAMESPACE_HEADER_START

template<typename Impl>
struct WriteCommand : public Command {
public:
    WriteCommand(const std::string& name, const std::string& description) : Command(name, description) {}
    ~WriteCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        Impl::run(args);
    }

    void postExecute() override {
    }

    void help() override {
        Impl::help(getName(), getDescription());
    }
};

WriteCommand<WriteSMTImpl>    writeSMTCmd("write_smt", "write smt");
WriteCommand<WriteAIGERImpl>  writeAIGERCmd("write_aiger", "write aiger");
WriteCommand<WriteBTOR2Impl>  writeBTOR2Cmd("write_btor2", "write btor2");

XUANSONG_NAMESPACE_HEADER_END


