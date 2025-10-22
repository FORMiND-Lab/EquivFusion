#include "infrastructure/base/command.h"
#include "infrastructure/log/log.h"
#include "infrastructure/tools/EquivMiterTool/equiv_miter_tool.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterCommand : public Command {
public:
    EquivMiterCommand() : Command("equiv_miter", "construct miter for logical equivalence check") {}
    ~EquivMiterCommand() override = default;
    
    void preExecute() override {
    }

    void execute(const std::vector<std::string> &args) override {
        // TODO(taomengxia: 20251022): Temp Code
        std::vector<std::string> argvStorage = {"equiv_miter"};
        argvStorage.insert(argvStorage.end(), args.begin(), args.end());
        
        std::vector<char*> argv;
        for (auto &str : argvStorage) {
            argv.push_back(str.data());
        }

        EquivMiterTool equivMiterTool;
        int result = equivMiterTool.run(argv.size(), argv.data());
        if (result != 0) {
            log("Error: equiv_miter failed");
        }
    }

    void postExecute() override {
    }

    void help() override {
        // TODO(taomengxia: 20251022): Temp code
        std::vector<std::string> args = {"--help"};
        execute(args);
    }
} equivMiterCommand;

XUANSONG_NAMESPACE_HEADER_END
