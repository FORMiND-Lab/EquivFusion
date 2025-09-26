#include "infrastructure/base/command.h"
#include "infrastructure/log/log.h"
#include "solving/run_solver.h"

XUANSONG_NAMESPACE_HEADER_START

struct SolverCommand : public Command {
public:
    SolverCommand() : Command("run_solver", "run solver") {}
    ~SolverCommand() override = default;

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
        
        if (solver.empty()) {
            logError("--solver is required");
            return;
        }
        if (inputFile.empty()) {
            logError("--inputfile is required");
            return;
        }
        XuanSong::RunSolver::runSolver(solver, inputFile, opts);
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("USAGE:    run_solver <--solver solver> <--inputfile file> [options]\n");
        log("OPTIONS:\n");
        log("   --solver <solver>\n");
        log("   --inputfile <file>\n");
        log("   --opts <options>\n");
        log("\n");
    }

} solverCommand;

XUANSONG_NAMESPACE_HEADER_END
