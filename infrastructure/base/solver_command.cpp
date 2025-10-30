#include "infrastructure/base/command.h"
#include "infrastructure/log/log.h"
#include "solving/solver-runner/solver_runner.h"

XUANSONG_NAMESPACE_HEADER_START

struct SolverCommand : public Command {
public:
    SolverCommand() : Command("solver_runner", "run solver") {}
    ~SolverCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        std::string solver = "";
        std::string inputFile = "";
        std::string opts = "";
        
        for (size_t argidx = 0; argidx < args.size(); argidx++) {
            if (args[argidx] == "--solver" && argidx + 1 < args.size()) {
                solver = args[++argidx].c_str();
            } else if (args[argidx] == "--inputfile" && argidx + 1 < args.size()) {
                inputFile = args[++argidx].c_str();
            } else if (args[argidx] == "--opts" && argidx + 1 < args.size()) {
                opts = args[++argidx].c_str();
            }
        }
        
        if (solver.empty() || inputFile.empty()) {
            log("Command solver_runner Failed:\n");
            if (solver.empty())     log("   --solver is required!\n");
            if (inputFile.empty())  log("   --inputfile is required!\n");
            return;
        }
        XuanSong::SolverRunner::run(solver, inputFile, opts);
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    solver_runner <--solver solver> <--inputfile file> [options]\n");
        log("   OPTIONS:\n");
        log("       --solver <solver>\n");
        log("       --inputfile <file>\n");
        log("       --opts <options>\n");
        log("\n");
    }

} solverCommand;

XUANSONG_NAMESPACE_HEADER_END
