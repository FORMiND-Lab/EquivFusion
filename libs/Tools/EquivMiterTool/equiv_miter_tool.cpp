#include "Tools/EquivMiterTool/equiv_miter_tool.h"

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Conversion/ExportAIGER.h"
#include "circt/Conversion/HWToBTOR2.h"

#include "circt/Dialect/Arc/ArcPasses.h"

#include "circt-passes/Miter/Passes.h"
#include "circt-passes/DecomposeConcat/Passes.h"

using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid
// conflict.
FailureOr<StringAttr> EquivMiterTool::mergeModules(ModuleOp dest, ModuleOp src,
                                                          StringAttr name) {
    SymbolTable destTable(dest), srcTable(src);
    StringAttr newName = {};
    for (auto &op : src.getOps()) {
        if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
            auto oldSymbol = symbol.getNameAttr();
            auto result = srcTable.renameToUnique(&op, {&destTable});
            if (failed(result))
                return src->emitError() << "failed to rename symbol " << oldSymbol;

            if (oldSymbol == name) {
                assert(!newName && "symbol must be unique");
                newName = *result;
            }
        }
    }

    if (!newName)
        return src->emitError()
               << "module " << name << " was not found in the second module";

    dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                           src.getBody()->getOperations());
    return newName;
}

// Parse one or two MLIR modules and merge it into a single module.
FailureOr<OwningOpRef<ModuleOp>> EquivMiterTool::parseAndMergeModules(MLIRContext &context, TimingScope &ts) {
    auto parserTimer = ts.nest("Parse and merge MLIR input(s)");

    if (inputFilenames.size() > 2) {
        llvm::errs() << "more than 2 files are provided!\n";
        return failure();
    }

    auto module = parseSourceFile<ModuleOp>(inputFilenames[0], &context);
    if (!module)
        return failure();

    if (inputFilenames.size() == 2) {
        auto moduleOpt = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
        if (!moduleOpt)
            return failure();
        auto result = mergeModules(module.get(), moduleOpt.get(),
                                   StringAttr::get(&context, secondModuleName));
        if (failed(result))
            return failure();

        secondModuleName.setValue(result->getValue().str());
    }

    return module;
}

LogicalResult EquivMiterTool::executeMiterToSMTLIB(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os) {
    pm.addPass(om::createStripOMPass());
    pm.addPass(emit::createStripEmitPass());

    pm.addPass(hw::createFlattenModules());
    EquivFusionMiterOptions opts = {firstModuleName, secondModuleName, EquivFusionMiter::MiterModeEnum::SMTLIB};
    pm.addPass(createEquivFusionMiter(opts));

    pm.addPass(createConvertHWToSMT());
    pm.addPass(createConvertCombToSMT());
    pm.addPass(createConvertVerifToSMT());

    pm.addPass(createSimpleCanonicalizerPass());

    if (failed(pm.run(module)))
        return failure();

    return smt::exportSMTLIB(module, os);
}


LogicalResult EquivMiterTool::executeMiterToAIGER(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os) {
    EquivFusionMiterOptions opts = {firstModuleName, secondModuleName, EquivFusionMiter::MiterModeEnum::AIGER};
    pm.addPass(createEquivFusionMiter(opts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createSimpleCanonicalizerPass());

    pm.nest<hw::HWModuleOp>().addPass(createConvertCombToAIG());
    pm.nest<hw::HWModuleOp>().addPass(aig::createLowerVariadic());

    if (failed(pm.run(module)))
        return failure();
    auto ops = module.getOps<hw::HWModuleOp>();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
        return failure();

    return aiger::exportAIGER(*ops.begin(), os);
}

LogicalResult EquivMiterTool::executeMiterToBTOR2(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os) {
    EquivFusionMiterOptions opts = {firstModuleName, secondModuleName, EquivFusionMiter::MiterModeEnum::BTOR2};
    pm.addPass(createEquivFusionMiter(opts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createEquivFusionDecomposeConcat());
    pm.addPass(arc::createSimplifyVariadicOpsPass());

    pm.addPass(createConvertHWToBTOR2Pass(os));

    return pm.run(module);
}

LogicalResult EquivMiterTool::executeMiter(MLIRContext &context, OwningOpRef<ModuleOp>& module, mlir::TimingScope &ts) {
    // Create the output directory or output file depending on our mode.
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    // Create an output file.
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }

    PassManager pm(&context);
    pm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(pm)))
        return failure();

    LogicalResult result = failure();
    switch (outputFormat) {
    case OutputSMTLIB:
        result = executeMiterToSMTLIB(pm, module.get(), outputFile.value()->os());
        break;
    case OutputAIGER:
        result = executeMiterToAIGER(pm, module.get(), outputFile.value()->os());
        break;
    case OutputBTOR2:
        result = executeMiterToBTOR2(pm, module.get(), outputFile.value()->os());
        break;
    }

    if (failed(result))
        return failure();

    outputFile.value()->keep();
    return success();
}


/// The entry point for the `equiv_miter` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeMiter` function to do the actual work.
int EquivMiterTool::run(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);

    // Hide default LLVM options, other than for this tool.
    // MLIR options are added below.
    cl::HideUnrelatedOptions(mainCategory);

    // Register any pass manager command line options.
    registerMLIRContextCLOptions();
    registerPassManagerCLOptions();
    registerDefaultTimingManagerCLOptions();
    registerAsmPrinterCLOptions();

    // Parse the command-line options provided by the user.
    cl::ParseCommandLineOptions(argc, argv, "equiv_miter");

    // Register the supported CIRCT dialects and create a context to work with.
    DialectRegistry registry;
    // clang-format off
    registry.insert<
      circt::comb::CombDialect,
      circt::emit::EmitDialect,
      circt::hw::HWDialect,
      circt::om::OMDialect,
      mlir::smt::SMTDialect,
      circt::verif::VerifDialect,
      mlir::func::FuncDialect
    >();
    // clang-format on
    mlir::func::registerInlinerExtension(registry);
    MLIRContext context(registry);

    // Create the timing manager we use to sample execution times.
    mlir::DefaultTimingManager tm;
    mlir::applyDefaultTimingManagerCLOptions(tm);
    auto ts = tm.getRootScope();

    // Parse and merge modules
    auto parsedModule = parseAndMergeModules(context, ts);
    if (failed(parsedModule)) {
        return -1;
    }
    OwningOpRef<ModuleOp> module = std::move(parsedModule.value());

    // Perform the logical equivalence checking
    LogicalResult result = executeMiter(context, module, ts);
    return failed(result) ? 1 : 0;
}

int EquivMiterTool::run(const std::vector<std::string> &args) {
    std::vector<std::string> argvStorage = {"equiv_miter"};
    argvStorage.insert(argvStorage.end(), args.begin(), args.end());

    std::vector<char*> argv;
    for (auto &str : argvStorage) {
        argv.push_back(str.data());
    }

    return run(argv.size(), argv.data());
}

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
