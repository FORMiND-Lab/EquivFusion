#include "infrastructure/base/command.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/log/log.h"

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileUtilities.h"

#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Transforms/Passes.h"

#include "circt-passes/FuncToHWModule/Passes.h"
#include "circt-passes/RemoveRedundantFunc/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

namespace {

struct ReadCCommandOptions {
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
            return;
        }

        return;
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    read_c [options] <--spec | --impl> <inputFile>\n");
        log("   OPTIONS:\n");
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
        if ((args[argidx] == "--top" || args[argidx] == "-top") && argidx + 1 < args.size()) {
            opts.topFunctionName = args[++argidx].c_str();
        } else if ((args[argidx] == "--spec" || args[argidx] == "-spec")) {
            opts.spec = true;
        } else if ((args[argidx] == "--impl" || args[argidx] == "-impl")) {
            opts.impl = true;
        } else {
            opts.inputFile = args[argidx].c_str();
        }
    }

    if (opts.inputFile.empty()) {
        log("Command 'read_c' Failed:\n");
        log("   Inputfile is required!\n");
        return false;
    }

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
    if (!opts.topFunctionName.empty()) {
        circt::EquivFusionRemoveRedundantFuncPassOptions options;
        options.topFunc = opts.topFunctionName;
        pm.addPass(circt::createEquivFusionRemoveRedundantFuncPass(options));
    }

    // Unroll affine loops.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopUnrollPass(-1, false, true, nullptr));
    
    // Lower affine to a conbination of Arith and
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLowerAffinePass());

    // Lower Scf to ControlFlow dialect.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCFToControlFlowPass());

    // Convert Arith to Comb
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(circt::createMapArithToCombPass());

    //Convert Func to Module
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(circt::createFuncToHWModule());
    
    pm.addPass(mlir::createCanonicalizerPass());
    
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
