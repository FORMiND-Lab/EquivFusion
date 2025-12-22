#include "infrastructure/base/command.h"
#include "infrastructure/utils/log-util/log_util.h"
#include "solving/solver-runner/solver_runner.h"

XUANSONG_NAMESPACE_HEADER_START

struct SolverCommandOptions {
    std::string solver = "";
    std::string inputFile = "";
    std::string options = "";
};

struct SolverCommand : public Command {
public:
    SolverCommand() : Command("solver_runner", "run solver") {}
    ~SolverCommand() = default;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        SolverCommandOptions opts;
        if (!parseOptions(args, opts)) {
            return;
        }

        XuanSong::SolverRunner::run(opts.solver, opts.inputFile, opts.options);
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
private:
    bool parseOptions(const std::vector<std::string>& args, SolverCommandOptions& opts);
} solverCommand;

bool SolverCommand::parseOptions(const std::vector<std::string>& args, SolverCommandOptions& opts) {
    for (size_t argidx = 0; argidx < args.size(); argidx++) {
        if (args[argidx] == "--solver" && argidx + 1 < args.size()) {
            opts.solver = args[++argidx];
        } else if (args[argidx] == "--inputfile" && argidx + 1 < args.size()) {
            opts.inputFile = args[++argidx];
        } else if (args[argidx] == "--opts" && argidx + 1 < args.size()) {
            opts.options = args[++argidx];
        }
    }

    if (opts.solver.empty() || opts.inputFile.empty()) {
        log("Command solver_runner Failed:\n");
        if (opts.solver.empty())     log("   --solver is required!\n");
        if (opts.inputFile.empty())  log("   --inputfile is required!\n");
        return false;
    }

    return true;
}

XUANSONG_NAMESPACE_HEADER_END
