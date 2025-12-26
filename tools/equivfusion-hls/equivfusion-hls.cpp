//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/FileUtilities.h"

#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombDialect.h"

#include "circt-passes/RemoveRedundantFunc/Passes.h"
#include "circt-passes/MemrefHLS/Passes.h"

#include "infrastructure/utils/hls-util/populate_hls_passes.h"

namespace cl = llvm::cl;


static cl::OptionCategory mainCategory("equivfusion-hls Options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("Input filename"), 
    cl::init("-"), cl::cat(mainCategory)
);

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
   cl::value_desc("filename"),
   cl::init("-"),
   cl::cat(mainCategory));

static cl::opt<std::string> topFunctionName("top", cl::desc("Top function name"),
   cl::value_desc("function name"),
   cl::init(""),
   cl::cat(mainCategory));

static cl::list<std::string> inputPorts("input-ports", cl::desc("Input ports"), 
   cl::cat(mainCategory), cl::value_desc("port name"));

static cl::list<std::string> outputPorts("output-ports", cl::desc("Output ports"), 
   cl::cat(mainCategory), cl::value_desc("port name"));

static llvm::LogicalResult 
processInput(mlir::MLIRContext &context, mlir::TimingScope &ts, 
             std::unique_ptr<llvm::MemoryBuffer> input, 
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
    
    context.allowUnregisteredDialects();

    // Setup of diagnostic handling.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
    mlir::SourceMgrDiagnosticHandler sourceMgrHandle(sourceMgr, &context);
    context.printOpOnDiagnostic(false);

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto parserTS = ts.nest("HLS Parse");
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        return llvm::failure();
    }

    // Apply any pass manager command line options.
    mlir::PassManager pm(&context);
    pm.enableTiming(ts);
    if (llvm::failed(mlir::applyPassManagerCLOptions(pm))) {
        return llvm::failure();
    }

    if (!topFunctionName.getValue().empty()) {
        circt::EquivFusionRemoveRedundantFuncPassOptions options;
        options.topFunc = topFunctionName.getValue();
        pm.addPass(circt::createEquivFusionRemoveRedundantFuncPass(options));
    }

    // Set direction of arguments for func.func.
    circt::EquivFusionSetFuncArgDirectionPassOptions options;
    options.inputPorts = llvm::SmallVector<std::string>(inputPorts.begin(), inputPorts.end());
    options.outputPorts = llvm::SmallVector<std::string>(outputPorts.begin(), outputPorts.end());
    pm.addNestedPass<mlir::func::FuncOp>(circt::createEquivFusionSetFuncArgDirectionPass(options));

    XuanSong::populateHLSPasses(pm);
    
    if (mlir::failed(pm.run(module.get()))) { 
        return llvm::failure();
    }

    module.get()->print((*outputFile)->os());

    // We intentionally "leak" the Module into the MLIRContext instead of
    // deallocating it.  There is no need to deallocate it right before process
    // exit.
    (void)module.release();
    return llvm::success();
}


static llvm::LogicalResult executeHls(mlir::MLIRContext &context) {
    // Create the timing manager we use to sample execution times.
    mlir::DefaultTimingManager tm;
    mlir::applyDefaultTimingManagerCLOptions(tm);
    auto ts = tm.getRootScope();

    // Set up the input file.
    std::string errorMessage;
    auto input = mlir::openInputFile(inputFilename, &errorMessage);
    if (!input) {
        llvm::errs() << errorMessage << "\n";
        return llvm::failure();
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    outputFile.emplace(mlir::openOutputFile(outputFilename, &errorMessage));
    if (!*outputFile) {
        llvm::errs() << errorMessage << "\n";
        return llvm::failure();
    }

    // Proecss the input
    if (llvm::failed(processInput(context, ts, std::move(input), outputFile))) {
        return llvm::failure();
    }

    // If the result succeeded and we're emitting a file, close it.
    if (outputFile.has_value()) {
        (*outputFile)->keep();
    }

    return llvm::success();
}


/// The entry point for the `equivfusion-hls` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeHLS` function to do the actual work.
int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);

    // Hide default LLVM options, other than for this tool.
    // MLIR options are added below.
    cl::HideUnrelatedOptions(mainCategory);

    // Register any pass manager command line options.
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    mlir::registerDefaultTimingManagerCLOptions();
    mlir::registerAsmPrinterCLOptions();

    // Parse the command-line options provided by the user.
    cl::ParseCommandLineOptions(argc, argv, "equivfusion-hls");
    
    // Register the supported CIRCT dialects and create a context to work with.
    mlir::DialectRegistry registry;

    // Register MLIR dialects.
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::scf::SCFDialect>();

    // Register MLIR passes.
    mlir::registerCSEPass();
    mlir::registerSCCPPass();
    mlir::registerInlinerPass();
    mlir::registerCanonicalizerPass();

    // Register CIRCT dialects.
    registry.insert<circt::hw::HWDialect, circt::comb::CombDialect, circt::seq::SeqDialect>();

    mlir::MLIRContext context(registry);

    auto result = executeHls(context);

    // Use "exit" instead of return'ing to signal completion.  This avoids
    // invoking the MLIRContext destructor, which spends a bunch of time
    // deallocating memory etc which process exit will do for us.
     exit(llvm::failed(result));
}



