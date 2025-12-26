//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/utils/path-util/path_util.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "libs/Tools/EquivMiter/equiv_miter.h"

#include "circt/Dialect/HW/HWOps.h"

#include "circt/Dialect/OM/OMPasses.h"              // createStripOMPass                CIRCTOMTransforms
#include "circt/Dialect/Emit/EmitPasses.h"          // createStripEmitPass              CIRCTEmitTransforms
#include "circt/Dialect/HW/HWPasses.h"              // createFlattenModules             CIRCTHWTransforms

#include "circt/Conversion/HWToSMT.h"               // createConvertHWToSMT             CIRCTHWToSMT
#include "circt/Conversion/CombToSMT.h"             // createConvertCombToSMT           CIRCTCombToSMT
#include "circt/Conversion/VerifToSMT.h"            // createConvertVerifToSMT          CIRCTVerifToSMT
#include "circt/Support/Passes.h"                   // createSimpleCanonicalizerPass    CIRCTSupport
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"        // exportSMTLIB                     MLIRExportSMTLIB

#include "circt/Conversion/CombToSynth.h"                   // createConvertCombToAIG   CIRCTCombToAIG
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"     // createLowerVariadic      CIRCTAIGTransforms
#include "circt/Conversion/ExportAIGER.h"                   // exportAIGER              CIRCTExportAIGER

#include "circt/Dialect/Arc/ArcPasses.h"            // createSimplifyVariadicOpsPass    CIRCTArcTransforms
#include "circt/Conversion/HWToBTOR2.h"             // createConvertHWToBTOR2Pass       CIRCTHWToBTOR2

#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include "circt-passes/CombTransforms/Passes.h"
#include "circt-passes/HWTransforms/Passes.h"
#include "circt-passes/Miter/Passes.h"              // createEquivFusionMiter           EquivFusionPassEquivMiter

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"


using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START
namespace EquivMiterTool {

struct EquivMiterToolOptions {
    bool printIR {false};
    circt::equivfusion::MiterModeEnum miterMode {circt::equivfusion::MiterModeEnum::SMTLIB};
    std::string specModuleName;
    std::string implModuleName;
    std::vector<std::string> inputFilenames;
    std::string outputFilename {"-"};
};

static bool parseOptions(const std::vector<std::string> &args, EquivMiterToolOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "--print-ir" || arg == "-print-ir") {
            opts.printIR = true;
        } else if ((arg == "-specModule" || arg == "--specModule") && idx + 1 < args.size()) {
            opts.specModuleName = args[++idx];
        } else if ((arg == "-implModule" || arg == "--implModule") && idx + 1 < args.size()) {
            opts.implModuleName = args[++idx];
        } else if (arg == "-o" && idx + 1 < args.size()) {
            opts.outputFilename = args[++idx];
            Utils::PathUtil::expandTilde(opts.outputFilename);
        } else if ((arg == "-mitermode" || arg == "--mitermode") && idx + 1 < args.size()) {
            auto val = args[++idx];
            if (val == "aiger") {
                opts.miterMode = circt::equivfusion::MiterModeEnum::AIGER;
            } else if (val == "btor2") {
                opts.miterMode = circt::equivfusion::MiterModeEnum::BTOR2;
            } else if (val == "smtlib") {
                opts.miterMode = circt::equivfusion::MiterModeEnum::SMTLIB;
            } else {
                log("Wrong option value of -mitermode.\n");
                return false;
            }
        }
    }

    if (opts.specModuleName.empty() || opts.implModuleName.empty()) {
        log("Both --specModule and --implModule must be specified.\n");
        return false;
    }

    return true;
}

static bool mergeModules(mlir::ModuleOp dest, ModuleOp src, EquivMiterToolOptions& opts, ModuleTypeEnum moduleType) {
    assert(moduleType == ModuleTypeEnum::SPEC || moduleType == ModuleTypeEnum::IMPL);

    bool isSpec = (moduleType == ModuleTypeEnum::SPEC);
    SymbolTable destTable(dest), srcTable(src);
    MLIRContext *context = EquivFusionManager::getInstance()->getGlobalContext();
    StringAttr moduleName = StringAttr::get(context, isSpec ? opts.specModuleName : opts.implModuleName);
    StringAttr newName = {};

    for (auto &op : src.getOps()) {
        if (SymbolOpInterface symbol = llvm::dyn_cast<SymbolOpInterface>(op)) {
            auto oldSymbol = symbol.getNameAttr();
            auto result = srcTable.renameToUnique(&op, {&destTable});

            if (llvm::failed(result)) { 
                log("[equiv_miter]: failed to rename symbol %s in %s\n", oldSymbol.getValue().str().c_str(), isSpec ? "specification" : "implementation");
                return false;
            }

            if (oldSymbol == moduleName) {
                if (newName) {
                    log("[equiv_miter]: module %s is not unique in %s\n", moduleName.getValue().str().c_str(), isSpec ? "specification" : "implementation");
                    return false;
                }
                newName = *result;
            }
        }
    }

    if (!newName) {
        log("[equiv_miter]: module %s is not found in %s\n", moduleName.getValue().str().c_str(), isSpec ? "specification" : "implementation");
        return false;
    }

    dest.getBody()->getOperations().splice(dest.getBody()->begin(), src.getBody()->getOperations());

    if (isSpec) {
        opts.specModuleName = newName.getValue().str();
    } else {
        opts.implModuleName = newName.getValue().str();
    }

    return true;
}

static void populatePreparePasses(mlir::PassManager& pm) {
    pm.addPass(om::createStripOMPass());
    pm.addPass(emit::createStripEmitPass());

    /// Inlines Private HW modules
    pm.addPass(hw::createFlattenModules());

    /// Flatten IO struct
    pm.addPass(circt::hw::createFlattenIO());
    pm.addPass(createSimpleCanonicalizerPass());

    /// Module Port array => integer
    pm.addPass(circt::equivfusion::hw::createEquivFusionFlattenIOArray());

    /// [Temp fix] Initialize post-defined operands for hw::ArrayInjectOp/hw::StructInjectOp
    pm.addPass(circt::equivfusion::hw::createEquivFusionInitPostDefinedOperands());
    /// [Temp fix]
    /// 1. hw.array_slice       => hw.array_get + hw.array_create
    /// 2. hw.struct_explode    => hw.struct_extract
    /// 3. hw.struct_inject     => hw.struct_extract + hw.struct_create
    pm.addPass(circt::equivfusion::hw::createEquivFusionHWAggregateOpsConvert());

    /// Aggregate Operations tp Comb operations
    pm.nest<hw::HWModuleOp>().addPass(hw::createHWAggregateToComb());
    /// [Temp fix]: Struct op to Comb
    pm.addPass(circt::equivfusion::hw::createEquivFusionHWAggregateToComb());

    /// Canonicalize
    pm.addPass(createSimpleCanonicalizerPass());
}

static llvm::LogicalResult executeMiterToSMT(mlir::PassManager &pm, mlir::ModuleOp module, llvm::raw_ostream &os,
                                             const circt::equivfusion::EquivFusionMiterOptions &miterOpts) {
    populatePreparePasses(pm);

    pm.addPass(circt::equivfusion::createEquivFusionMiter(miterOpts));

    pm.addPass(circt::createConvertHWToSMT());
    pm.addPass(circt::createConvertCombToSMT());
    pm.addPass(circt::createConvertVerifToSMT());
    pm.addPass(circt::createSimpleCanonicalizerPass());

    if (failed(pm.run(module)))
        return failure();

    return smt::exportSMTLIB(module, os);
}

static llvm::LogicalResult executeMiterToAIGER(mlir::PassManager &pm, mlir::ModuleOp module, llvm::raw_ostream &os,
                                               const circt::equivfusion::EquivFusionMiterOptions &miterOpts) {
    populatePreparePasses(pm);

    pm.addPass(circt::equivfusion::createEquivFusionMiter(miterOpts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createSimpleCanonicalizerPass());

    ConvertCombToSynthOptions options;
    options.targetIR = CombToSynthTargetIR::AIG;
    pm.nest<hw::HWModuleOp>().addPass(createConvertCombToSynth(options));
    pm.nest<hw::HWModuleOp>().addPass(synth::createLowerVariadic());

    if (failed(pm.run(module)))
        return failure();
    auto ops = module.getOps<hw::HWModuleOp>();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
        return failure();

    circt::aiger::ExportAIGEROptions exportAIGEROpts = {true, true};
    return aiger::exportAIGER(*ops.begin(), os, &exportAIGEROpts);
}

static LogicalResult executeMiterToBTOR2(mlir::PassManager &pm, mlir::ModuleOp module, llvm::raw_ostream &os,
                                         const circt::equivfusion::EquivFusionMiterOptions &miterOpts) {
    populatePreparePasses(pm);

    pm.addPass(circt::equivfusion::createEquivFusionMiter(miterOpts));

    pm.addPass(circt::hw::createFlattenModules());
    pm.addPass(createSimpleCanonicalizerPass());

    pm.addPass(createConvertHWToBTOR2Pass(os));
    return pm.run(module);
}

void help(const std::string &name, const std::string &description) {
    log("\n");
    log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());;
    log("   USAGE:    %s <--specModule name> <--implModule name> [options]\n", name.c_str());
    log("   OPTIONS:\n");
    log("       --print-ir ----------------------------- Print IR after pass\n");
    log("       --specModule <module name> ------------- Specify a named module for the specification circuit\n");
    log("       --implModule <module name> ------------- Specify a named module for the implementation circuit\n");
    log("       --mitermode ---------------------------- MiterMode [smtlib, aiger, btor2], default is smtlib\n");
    log("       -o ------------------------------------- Output filename\n");
    log("\n\n");
}

bool execute(const std::vector<std::string>& args) {
    EquivMiterToolOptions opts;
    if (!parseOptions(args, opts)) {
        log("[equiv_miter]: parse options failed\n\n");
        return false;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[equiv_miter]: opend output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    LogicalResult result = failure();
    MLIRContext *context = EquivFusionManager::getInstance()->getGlobalContext();
    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(context));
    ModuleOp specModule = EquivFusionManager::getInstance()->getSpecModuleOp();
    ModuleOp implModule = EquivFusionManager::getInstance()->getImplModuleOp();


    if (!specModule) {
        log("[equiv_miter]: Specification not specified, please use 'read_c' or 'read_v' to read the specification before 'equiv_miter'.\n");
        return false;
    }

    if (!implModule) {
        log("[equiv_miter]: Implementation not specified, please use 'read_c' or 'read_v' to read the implementation before 'equiv_miter'.\n");
        return false;
    }

    if (!mergeModules(module.get(), specModule, opts, ModuleTypeEnum::SPEC)) {
        return false;
    }

    if (!mergeModules(module.get(), implModule, opts, ModuleTypeEnum::IMPL)) {
        return false;
    }

    PassManager pm(context);
    EquivFusionManager::getInstance()->configureIRPrinting(pm, opts.printIR);

    circt::equivfusion::EquivFusionMiterOptions miterOpts = {opts.specModuleName, opts.implModuleName, opts.miterMode};
    switch (opts.miterMode) {
        case circt::equivfusion::MiterModeEnum::SMTLIB:
            result = executeMiterToSMT(pm, module.get(), outputFile.value()->os(), miterOpts);
            break;
        case circt::equivfusion::MiterModeEnum::AIGER:
            result = executeMiterToAIGER(pm, module.get(), outputFile.value()->os(), miterOpts);
            break;
        case circt::equivfusion::MiterModeEnum::BTOR2:
            result = executeMiterToBTOR2(pm, module.get(), outputFile.value()->os(), miterOpts);
            break;
    }

    if (failed(result)) {
        log("[equiv_miter]: miter failed\n\n");
        return false;
    }

    outputFile.value()->keep();

    OwningOpRef<ModuleOp> specNull, implNull;

    EquivFusionManager::getInstance()->setSpecModuleOp(specNull);
    EquivFusionManager::getInstance()->setImplModuleOp(implNull);
    EquivFusionManager::getInstance()->setMergedModuleOp(module);

    return true;
}

}  // namespace EquivMiterTool
XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
