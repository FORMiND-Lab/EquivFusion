#include "infrastructure/log/log.h"
#include "libs/Tools/EquivMiter/equiv_miter.h"

#include "mlir/Pass/PassManager.h"

#include "circt/Dialect/OM/OMPasses.h"      // createStripOMPass                CIRCTOMTransforms
#include "circt/Dialect/Emit/EmitPasses.h"  // createStripEmitPass              CIRCTEmitTransforms
#include "circt/Dialect/HW/HWPasses.h"      // createFlattenModules             CIRCTHWTransforms
#include "circt-passes/Miter/Passes.h"      // createEquivFusionMiter           EquivFusionPassEquivMiter

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START


bool EquivMiterImpl::run(const std::vector<std::string>& args, mlir::MLIRContext& context,
                         mlir::ModuleOp module, mlir::OwningOpRef<mlir::ModuleOp> &outputModule) {
    EquivMiterImplOptions opts;
    if (!parserOptions(args, opts)) {
        log("[equiv_miter]: parser options failed\n\n");
        return false;
    }    

    PassManager pm(&context);
    
    EquivFusionMiterOptions miterOpts = {opts.firstModuleName, opts.secondModuleName, opts.miterMode};

    switch (opts.miterMode) {
    case EquivFusionMiter::MiterModeEnum::SMTLIB:
        pm.addPass(om::createStripOMPass());
        pm.addPass(emit::createStripEmitPass());
        pm.addPass(hw::createFlattenModules());
        pm.addPass(createEquivFusionMiter(miterOpts));
        break;
    case EquivFusionMiter::MiterModeEnum::AIGER:
        pm.addPass(createEquivFusionMiter(miterOpts));
        break;
    case EquivFusionMiter::MiterModeEnum::BTOR2:
        pm.addPass(createEquivFusionMiter(miterOpts));
        break;
    }

    if (failed(pm.run(module))) {
        log("[equiv_miter]: run PassManager failed\n\n");
        return false;
    }

    if (!opts.outputFilename.empty() && module) {
        std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
        std::string errorMessage;
        outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
        if (!outputFile.value()) {
            llvm::errs() << errorMessage << "\n";
            return false;
        }
        module.print(outputFile.value()->os());
        outputFile.value()->keep();   
    }

    outputModule = module.clone();
    return true;
}

bool EquivMiterImpl::parserOptions(const std::vector<std::string> &args, EquivMiterImplOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "--c1" && idx + 1 < args.size()) {
            opts.firstModuleName = args[++idx];
        } else if (arg == "--c2" && idx + 1 < args.size()) {
            opts.secondModuleName = args[++idx];
        } else if (arg == "-o" && idx + 1 < args.size()) {
            opts.outputFilename = args[++idx];
        } else if (arg == "--mitermode" && idx + 1 < args.size()) {
            auto val = args[++idx];
            if (val == "aiger") {
                opts.miterMode = EquivFusionMiter::MiterModeEnum::AIGER;
            } else if (val == "btor2") {
                opts.miterMode = EquivFusionMiter::MiterModeEnum::BTOR2;
            } else if (val == "smtlib") {
                opts.miterMode = EquivFusionMiter::MiterModeEnum::SMTLIB;
            } else {
                log("Wrong option value of --mitermode.\n");
                return false;
            }
        }
    }    

    if (opts.firstModuleName.empty() || opts.secondModuleName.empty()) {
        log("Both --c1 and --c2 must be specified.\n");
        return false;
    }

    return true;
}

void EquivMiterImpl::help(const std::string& name, const std::string& description) {
    log("\n");
    log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());;
    log("   USAGE:    %s <--c1 name1> <--c2 name2> <inputfile1 [inputfile2]> [options]\n", name.c_str());
    log("   OPTIONS:\n");
    log("       --c1 <module name>      - Specify a named module for the first circuit of the comparison\n");
    log("       --c2 <module name>      - Specify a named module for the second circuit of the comparison\n");
    log("       --mitermode             - MiterMode [smtlib, aiger, btor2], default is smtlib\n");
    log("   Example:");
    log("       equiv_miter --c1 mod1 --c2 mod2 file1.mlir file2.mlir");
    log("\n\n");
}

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
