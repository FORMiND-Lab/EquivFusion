#include "infrastructure/base/command.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/utils/path-util/path_util.h"
#include "infrastructure/utils/hls-util/populate_hls_passes.h"

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "circt/Conversion/Passes.h"
#include "circt/Transforms/Passes.h"

#include "circt-passes/RemoveRedundantFunc/Passes.h"
#include "circt-passes/MemrefHLS/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

namespace {

struct ReadCCommandOptions {
    bool printIR = false;
    std::string topFunctionName = "";
    std::string inputFile = "";
    bool spec = false;
    bool impl = false;
};

} // namespace

struct ReadCCommand : public Command {
    ReadCCommand() : Command("read_c", "read c file") {}
    ~ReadCCommand() = default;

    ReadCCommand(const ReadCCommand &) = delete;
    ReadCCommand &operator=(const ReadCCommand &) = delete;
    ReadCCommand(ReadCCommand &&) = delete;
    ReadCCommand &operator=(ReadCCommand &&) = delete;

    bool parseOptions(const std::vector<std::string>& args, ReadCCommandOptions& opts);
    llvm::LogicalResult executeHLS(const ReadCCommandOptions& opts);
    llvm::LogicalResult executeCFlow(const ReadCCommandOptions& opts);

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override { 
        ReadCCommandOptions opts;
        if (!parseOptions(args, opts)) {
            return;
        }

        if (llvm::failed(executeCFlow(opts))) {
            logError("Command 'read_c' Failed!\n");
            EquivFusionManager::getInstance()->clearPorts();
            return;
        }

        EquivFusionManager::getInstance()->clearPorts();
        return;
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    read_c [options] <--spec | --impl> <inputFile>\n");
        log("   OPTIONS:\n");
        log("       --print-ir ----------------------------- Print IR after pass\n");
        log("       --top <topFunctionName> ---------------- Specify top function name\n");
        log("       --spec --------------------------------- Design is specification\n");
        log("       --impl --------------------------------- Design is implementation\n");
        log("\n");
    }
} readCCommand;


bool ReadCCommand::parseOptions(const std::vector<std::string>& args, ReadCCommandOptions& opts) { 
    mlir::ModuleOp specModule = EquivFusionManager::getInstance()->getSpecModuleOp();
    mlir::ModuleOp implModule = EquivFusionManager::getInstance()->getImplModuleOp();
        
    for (size_t argidx = 0; argidx < args.size(); argidx++) {
        if (args[argidx] == "--print-ir" || args[argidx] == "-print-ir") {
            opts.printIR = true;
        } else if ((args[argidx] == "--top" || args[argidx] == "-top") && argidx + 1 < args.size()) {
            opts.topFunctionName = args[++argidx].c_str();
        } else if ((args[argidx] == "--spec" || args[argidx] == "-spec")) {
            opts.spec = true;
        } else if ((args[argidx] == "--impl" || args[argidx] == "-impl")) {
            opts.impl = true;
        } else {
            opts.inputFile = args[argidx].c_str();
            Utils::PathUtil::expandTilde(opts.inputFile);
        }
    }

    if (opts.inputFile.empty()) {
        log("Command 'read_c' Failed:\n");
        log("   Inputfile is required!\n");
        return false;
    }

    Utils::PathUtil::expandTilde(opts.inputFile);

    if (opts.spec && opts.impl) {
        log("Command 'read_c' Failed:\n");
        log("   --spec and --impl cannot be specified at the same time!\n");
        return false;
    } else if (!opts.spec && !opts.impl) {
        log("Command 'read_c' Failed:\n");
        log("   --spec or --impl must be specified!\n");
        return false;
    } else if (opts.spec) {
        if (specModule) {
            log("Command 'read_c' Failed:\n");
            log("   Specification design already specified!\n");
            return false;
        }
    } else if (opts.impl) {
        if (implModule) {
            log("Command 'read_c' Failed:\n");
            log("   Implementation design already specified!\n");
            return false;
        }
    }    

    return true;
}

llvm::LogicalResult ReadCCommand::executeHLS(const ReadCCommandOptions& opts) {
    mlir::MLIRContext &context = *EquivFusionManager::getInstance()->getGlobalContext();
    mlir::OwningOpRef<mlir::ModuleOp> module;

    std::string errorMessage;
    auto input = mlir::openInputFile(opts.inputFile, &errorMessage);
    if (!input) {
        llvm::errs() << errorMessage << "\n";
        return llvm::failure();
    }

    module = mlir::parseSourceFile<mlir::ModuleOp>(opts.inputFile, &context);
    if (!module) {
        return llvm::failure();
    }

    mlir::PassManager pm(&context);
    EquivFusionManager::getInstance()->configureIRPrinting(pm, opts.printIR);

    if (!opts.topFunctionName.empty()) {
        circt::EquivFusionRemoveRedundantFuncPassOptions options;
        options.topFunc = opts.topFunctionName;
        pm.addPass(circt::createEquivFusionRemoveRedundantFuncPass(options));
    }

    // Set direction of arguments for func.func.
    circt::EquivFusionSetFuncArgDirectionPassOptions options;
    std::set<std::string> inputPorts = EquivFusionManager::getInstance()->getInputPorts();
    std::set<std::string> outputPorts = EquivFusionManager::getInstance()->getOutputPorts();

    options.inputPorts = llvm::SmallVector<std::string>(inputPorts.begin(), inputPorts.end());
    options.outputPorts = llvm::SmallVector<std::string>(outputPorts.begin(), outputPorts.end());
    pm.addNestedPass<mlir::func::FuncOp>(circt::createEquivFusionSetFuncArgDirectionPass(options));

    populateHLSPasses(pm);
    
    if (mlir::failed(pm.run(module.get()))) { 
        return llvm::failure();
    }

    if (opts.spec) {
        EquivFusionManager::getInstance()->setSpecModuleOp(module);
    } else {
        EquivFusionManager::getInstance()->setImplModuleOp(module);
    }

    return llvm::success();
}

llvm::LogicalResult ReadCCommand::executeCFlow(const ReadCCommandOptions& opts) {
    

    if (llvm::failed(executeHLS(opts))) {
        return llvm::failure();
    }
/*
    // Debug Code.

    if (opts.spec) {
        EquivFusionManager::getInstance()->getSpecModuleOp().print(llvm::outs());
    } else {
        EquivFusionManager::getInstance()->getImplModuleOp().print(llvm::outs());
    }
*/
    return llvm::success();
}










XUANSONG_NAMESPACE_HEADER_END
