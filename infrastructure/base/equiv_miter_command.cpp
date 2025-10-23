#include "infrastructure/base/command.h"
#include "infrastructure/log/log.h"
#include "Tools/EquivMiterTool/equiv_miter_tool.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterCommand : public Command {
public:
    EquivMiterCommand() : Command("equiv_miter", "construct miter for logical equivalence check") {}
    ~EquivMiterCommand() override = default;
    
    void preExecute() override {
    }

    void execute(const std::vector<std::string> &args) override {
        EquivMiterTool equivMiterTool;
        int result = equivMiterTool.run(args);
        if (result != 0) {
            log("[Command Failed]: equiv_miter failed\n\n");
        }
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());;
        log("   USAGE:    equiv_miter <--c1 name1> <--c2 name2> <inputfile1 [inputfile2]> [options]\n");
        log("   OPTIONS:\n");
        log("       --c1 <module name>      - Specify a named module for the first circuit of the comparison\n");
        log("       --c2 <module name>      - Specify a named module for the second circuit of the comparison\n");
        log("       -o <filename>           - Output filename\n");
        log("       Specify output format\n");
        log("           --smtlib                - smt object file [default]\n");
        log("           --aiger                 - aiger object file\n");
        log("           --btor2                 - btor2 object file\n");
        log("   Example:");
        log("       equiv_miter --c1 mod1 --c2 mod2 file1.mlir file2.mlir");
        log("\n\n");
    }
} equivMiterCommand;

XUANSONG_NAMESPACE_HEADER_END
