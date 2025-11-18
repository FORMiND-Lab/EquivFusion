#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/base/command.h"
#include "circt-passes/TemporalUnroll/Passes.h"

#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

struct UnrollOptions {
    unsigned steps = 1;
    ModuleType moduleType {ModuleType::UNKNOWN};
};

struct UnrollCommand : public Command {
    UnrollCommand() : Command("unroll", "temporal unroll") {}
    ~UnrollCommand() = default;

    UnrollCommand(const UnrollCommand &) = delete;
    UnrollCommand &operator=(const UnrollCommand &) = delete;
    UnrollCommand(UnrollCommand &&) = delete;
    UnrollCommand &operator=(UnrollCommand &&) = delete;

    void preExecute() override {
    }

    void execute(const std::vector<std::string> &args) override {
        UnrollOptions opts;
        if (!parseOptions(args, opts)) {
            return;
        }

        if (llvm::failed(executeUnroll(opts))) {
            logError("[unroll]: failed\n");
        }
    }

    void postExecute() override {
    }
    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    unroll [options] <--spec | --impl | --miter>\n");
        log("   OPTIONS:\n");
        log("       --spec --------------------------------- specification module\n");
        log("       --impl --------------------------------- implementation module\n");
        log("       --miter -------------------------------- miter module\n");
        log("       --steps -------------------------------- unroll steps\n");
        log("\n");
    }

private:
    bool parseOptions(const std::vector<std::string> &args, UnrollOptions &opts);
    llvm::LogicalResult executeUnroll(const UnrollOptions &opts);
} unrollCmd;

bool UnrollCommand::parseOptions(const std::vector<std::string> &args, UnrollOptions &opts) {
    for (size_t idx = 0; idx < args.size(); ++idx) {
        if ((args[idx] == "--steps" || args[idx] == "-steps") && (idx + 1 < args.size())) {
            opts.steps = std::stoi(args[++idx]);
        } if (args[idx] == "--spec" || args[idx] == "-spec") {
            opts.moduleType = ModuleType::SPEC;
        } else if (args[idx] == "--impl" || args[idx] == "-impl") {
            opts.moduleType = ModuleType::IMPL;
        } else if (args[idx] == "--miter" || args[idx] == "-miter") {
            opts.moduleType = ModuleType::MITER;
        }
    }
    if (opts.moduleType == ModuleType::UNKNOWN) {
        log("[unroll]: please specify module type.\n");
        return false;
    }

    return true;
}

llvm::LogicalResult UnrollCommand::executeUnroll(const UnrollOptions &opts) {
    if (opts.steps <= 1) {
        logWarning("[unroll]: no need unroll, steps[%d] <= 1\n", opts.steps);
        return llvm::success();
    }

    mlir::ModuleOp module = EquivFusionManager::getInstance()->getModuleOp(opts.moduleType);
    if (!module) {
        logWarning("[unroll]: no need unroll, module is empty.\n");
        return llvm::success();
    }

    mlir::OwningOpRef<mlir::ModuleOp> clonedModule = module.clone();
    mlir::MLIRContext *context = EquivFusionManager::getInstance()->getGlobalContext();
    mlir::DefaultTimingManager tm;
    auto ts = tm.getRootScope();

    mlir::PassManager pm(context);
    pm.enableVerifier(true);
    pm.enableTiming(ts);

    circt::EquivFusionTemporalUnrollOptions timeUnrollOpts = {opts.steps};
    pm.addPass(circt::createEquivFusionTemporalUnroll(timeUnrollOpts));
#if 0
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
#endif

    if (mlir::failed(pm.run(clonedModule.get()))) {
        log("[unroll]: Failed to unroll.\n");
        return llvm::failure();
    }

    EquivFusionManager::getInstance()->setModuleOp(clonedModule, opts.moduleType);
    return llvm::success();
}

XUANSONG_NAMESPACE_HEADER_END