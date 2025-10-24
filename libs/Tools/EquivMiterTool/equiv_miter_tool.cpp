#include "libs/Tools/EquivMiterTool/equiv_miter_tool.h"
#include "infrastructure/log/log.h"

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

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid conflict.
FailureOr<StringAttr> EquivMiterTool::mergeModules(ModuleOp dest, ModuleOp src, StringAttr name) {
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
FailureOr<OwningOpRef<ModuleOp>> EquivMiterTool::parseAndMergeModules(MLIRContext &context) {
    auto module = parseSourceFile<ModuleOp>(options_.inputFilenames[0], &context);
    if (!module)
        return failure();

    if (options_.inputFilenames.size() == 2) {
        auto moduleOpt = parseSourceFile<ModuleOp>(options_.inputFilenames[1], &context);
        if (!moduleOpt)
            return failure();
        auto result = mergeModules(module.get(), moduleOpt.get(),
                                   StringAttr::get(&context, options_.secondModuleName));
        if (failed(result))
            return failure();
        
        options_.secondModuleName = result->getValue().str();
    }

    return module;
}

LogicalResult EquivMiterTool::executeMiterToSMTLIB(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os) {
    pm.addPass(om::createStripOMPass());
    pm.addPass(emit::createStripEmitPass());

    pm.addPass(hw::createFlattenModules());
    
    EquivFusionMiterOptions opts = {options_.firstModuleName, options_.secondModuleName, options_.miterMode};
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
    EquivFusionMiterOptions opts = {options_.firstModuleName, options_.secondModuleName, options_.miterMode};
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
    EquivFusionMiterOptions opts = {options_.firstModuleName, options_.secondModuleName, options_.miterMode};
    pm.addPass(createEquivFusionMiter(opts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createEquivFusionDecomposeConcat());
    pm.addPass(arc::createSimplifyVariadicOpsPass());

    pm.addPass(createConvertHWToBTOR2Pass(os));

    return pm.run(module);
}

LogicalResult EquivMiterTool::executeMiter(MLIRContext &context, OwningOpRef<ModuleOp>& module) {
    // Create the output directory or output file depending on our mode.
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    // Create an output file.
    outputFile.emplace(openOutputFile(options_.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }

    PassManager pm(&context);

    LogicalResult result = failure();
    switch (options_.miterMode) {
    case EquivFusionMiter::MiterModeEnum::SMTLIB:
        result = executeMiterToSMTLIB(pm, module.get(), outputFile.value()->os());
        break;
    case EquivFusionMiter::MiterModeEnum::AIGER:
        result = executeMiterToAIGER(pm, module.get(), outputFile.value()->os());
        break;
    case EquivFusionMiter::MiterModeEnum::BTOR2:
        result = executeMiterToBTOR2(pm, module.get(), outputFile.value()->os());
        break;
    }

    if (failed(result))
        return failure();

    outputFile.value()->keep();
    return success();
}


/// The entry point for the `equiv_miter` tool
bool EquivMiterTool::run() {
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

    // Parse and merge modules
    auto parsedModule = parseAndMergeModules(context);
    if (failed(parsedModule)) {
        return -1;
    }
    OwningOpRef<ModuleOp> module = std::move(parsedModule.value());

    // Perform the logical equivalence checking
    LogicalResult result = executeMiter(context, module);
    return failed(result) ? false : true;
}

bool EquivMiterTool::initOptions(const std::vector<std::string> &args) {
    std::vector<std::string> remainingArgv;

    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "--c1" && idx + 1 < args.size()) {
            options_.firstModuleName = args[++idx];
        } else if (arg == "--c2" && idx + 1 < args.size()) {
            options_.secondModuleName = args[++idx];
        } else if (arg == "-o" && idx + 1 < args.size()) {
            options_.outputFilename = args[++idx];
        } else if (arg == "--mitermode" && idx + 1 < args.size()) {
            auto val = args[++idx];
            if (val == "aiger") {
                options_.miterMode = EquivFusionMiter::MiterModeEnum::AIGER;
            } else if (val == "btor2") {
                options_.miterMode = EquivFusionMiter::MiterModeEnum::BTOR2;
            } else if (val == "smtlib") {
                options_.miterMode = EquivFusionMiter::MiterModeEnum::SMTLIB;
            } else {
                log("Wrong option value of --mitermode.\n");
                return false;
            }
        } else {
            options_.inputFilenames.push_back(arg);
        }
    }    

    if (options_.firstModuleName.empty() || options_.secondModuleName.empty()) {
        log("Both --c1 and --c2 must be specified.\n");
        return false;
    }

    if (options_.inputFilenames.empty() || options_.inputFilenames.size() > 2) {
        log("Must provide 1 or 2 input files.\n");
        return false;
    }

    return true;
}

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
