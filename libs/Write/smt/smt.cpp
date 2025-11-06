#include "infrastructure/utils/log/log.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"

#include "libs/Write/smt/smt.h"

#include "mlir/Target/SMTLIB/ExportSMTLIB.h"  // exportSMTLIB                       MLIRExportSMTLIB

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteSMTImpl::run(const std::vector<std::string>& args) {
    // Parse Options
    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_smt]: parse options failed\n\n");
        return false;
    }

    // Check module
    mlir::ModuleOp module = EquivFusionManager::getInstance()->getMergedModuleOp();
    if (!module) {
        return true;
    }

    // Open output file
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_smt]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    // Print SMT to output file
    if (failed(mlir::smt::exportSMTLIB(module, outputFile.value()->os()))) {
        log("[write_smt]: exportSMTLIB failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
