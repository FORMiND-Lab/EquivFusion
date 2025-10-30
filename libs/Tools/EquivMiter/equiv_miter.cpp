#include "infrastructure/log/log.h"
#include "libs/Tools/EquivMiter/equiv_miter.h"

#include "circt/Dialect/HW/HWOps.h"

#include "circt/Dialect/OM/OMPasses.h"              // createStripOMPass                CIRCTOMTransforms
#include "circt/Dialect/Emit/EmitPasses.h"          // createStripEmitPass              CIRCTEmitTransforms
#include "circt/Dialect/HW/HWPasses.h"              // createFlattenModules             CIRCTHWTransforms
#include "circt-passes/Miter/Passes.h"              // createEquivFusionMiter           EquivFusionPassEquivMiter
#include "circt/Conversion/HWToSMT.h"               // createConvertHWToSMT             CIRCTHWToSMT
#include "circt/Conversion/CombToSMT.h"             // createConvertCombToSMT           CIRCTCombToSMT
#include "circt/Conversion/VerifToSMT.h"            // createConvertVerifToSMT          CIRCTVerifToSMT
#include "circt/Support/Passes.h"                   // createSimpleCanonicalizerPass    CIRCTSupport
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"        // exportSMTLIB                     MLIRExportSMTLIB

#include "circt/Conversion/CombToAIG.h"             // createConvertCombToAIG           CIRCTCombToAIG
#include "circt/Dialect/AIG/AIGPasses.h"            // createLowerVariadic              CIRCTAIGTransforms
#include "circt/Conversion/ExportAIGER.h"           // exportAIGER                      CIRCTExportAIGER

#include "circt-passes/DecomposeConcat/Passes.h"    // createEquivFusionDecomposeConcat EquivFusionPassDecomposeConcat
#include "circt/Dialect/Arc/ArcPasses.h"            // createSimplifyVariadicOpsPass    CIRCTArcTransforms
#include "circt/Conversion/HWToBTOR2.h"             // createConvertHWToBTOR2Pass       CIRCTHWToBTOR2

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START

bool EquivMiterTool::run(const std::vector<std::string>& args, mlir::MLIRContext& context,
                         mlir::ModuleOp module, mlir::OwningOpRef<mlir::ModuleOp> &outputModule) {
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

    EquivFusionMiterOptions miterOpts = {opts.firstModuleName, opts.secondModuleName, opts.miterMode};
    LogicalResult result = failure();
    
    PassManager pm(&context);
    switch (opts.miterMode) {
    case EquivFusionMiter::MiterModeEnum::SMTLIB:
        result = miterToSMT(pm, module, outputFile.value()->os(), miterOpts);
        break;
    case EquivFusionMiter::MiterModeEnum::AIGER:
        result = miterToAIGER(pm, module, outputFile.value()->os(), miterOpts);
        break;
    case EquivFusionMiter::MiterModeEnum::BTOR2:
        result = miterToBTOR2(pm, module, outputFile.value()->os(), miterOpts);
        break;
    }

    if (failed(result)) {
        log("[equiv_miter]: miter failed\n\n");
        return false;
    }

    outputFile.value()->keep();

    outputModule = module.clone();
    return true;
}

llvm::LogicalResult EquivMiterTool::miterToSMT(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                               const EquivFusionMiterOptions& miterOpts) {
    pm.addPass(om::createStripOMPass());
    pm.addPass(emit::createStripEmitPass());
    pm.addPass(hw::createFlattenModules());
    pm.addPass(createEquivFusionMiter(miterOpts));

    pm.addPass(circt::createConvertHWToSMT());
    pm.addPass(circt::createConvertCombToSMT());
    pm.addPass(circt::createConvertVerifToSMT());
    pm.addPass(circt::createSimpleCanonicalizerPass());

    if (failed(pm.run(module)))
        return failure();

    return smt::exportSMTLIB(module, os);
}

llvm::LogicalResult EquivMiterTool::miterToAIGER(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                                 const EquivFusionMiterOptions& miterOpts) {
    pm.addPass(createEquivFusionMiter(miterOpts));
    
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

LogicalResult EquivMiterTool::miterToBTOR2(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                               const EquivFusionMiterOptions& miterOpts) {
    pm.addPass(createEquivFusionMiter(miterOpts));

    pm.addPass(hw::createFlattenModules());
    pm.addPass(createEquivFusionDecomposeConcat());
    pm.addPass(arc::createSimplifyVariadicOpsPass());

    pm.addPass(createConvertHWToBTOR2Pass(os));

    return pm.run(module);   
}

bool EquivMiterTool::parseOptions(const std::vector<std::string> &args, EquivMiterToolOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "-c1" && idx + 1 < args.size()) {
            opts.firstModuleName = args[++idx];
        } else if (arg == "-c2" && idx + 1 < args.size()) {
            opts.secondModuleName = args[++idx];
        } else if (arg == "-o" && idx + 1 < args.size()) {
            opts.outputFilename = args[++idx];
        } else if (arg == "-mitermode" && idx + 1 < args.size()) {
            auto val = args[++idx];
            if (val == "aiger") {
                opts.miterMode = EquivFusionMiter::MiterModeEnum::AIGER;
            } else if (val == "btor2") {
                opts.miterMode = EquivFusionMiter::MiterModeEnum::BTOR2;
            } else if (val == "smtlib") {
                opts.miterMode = EquivFusionMiter::MiterModeEnum::SMTLIB;
            } else {
                log("Wrong option value of -mitermode.\n");
                return false;
            }
        }
    }    

    if (opts.firstModuleName.empty() || opts.secondModuleName.empty()) {
        log("Both -c1 and -c2 must be specified.\n");
        return false;
    }

    return true;
}

void EquivMiterTool::help(const std::string& name, const std::string& description) {
    log("\n");
    log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());;
    log("   USAGE:    %s <-c1 name1> <-c2 name2> [options]\n", name.c_str());
    log("   OPTIONS:\n");
    log("       -c1 <module name>      - Specify a named module for the first circuit of the comparison\n");
    log("       -c2 <module name>      - Specify a named module for the second circuit of the comparison\n");
    log("       -mitermode             - MiterMode [smtlib, aiger, btor2], default is smtlib\n");
    log("       -o                     - Output filename\n");
    log("   Example:");
    log("       equiv_miter -c1 mod1 -c2 mod2");
    log("\n\n");
}

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
