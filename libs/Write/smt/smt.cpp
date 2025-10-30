#include "libs/Write/smt/smt.h"

#include "mlir/Target/SMTLIB/ExportSMTLIB.h"  // exportSMTLIB                       MLIRExportSMTLIB

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteSMTImpl::run(const std::vector<std::string>& args, mlir::MLIRContext &context, mlir::ModuleOp inputModule) {
    if (!inputModule) {
        return true;
    }    

    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_smt]: parse options failed\n\n");
        return false;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_smt]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }
    
    if (failed(mlir::smt::exportSMTLIB(inputModule, outputFile.value()->os()))) {
        log("[write_smt]: exportSMTLIB failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
