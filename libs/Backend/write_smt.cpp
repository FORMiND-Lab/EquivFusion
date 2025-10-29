#include "libs/Backend/write_smt.h"

#include "mlir/Dialect/SMT/IR/SMTDialect.h"

#include "circt/Conversion/HWToSMT.h"         // createConvertHWToSMT               CIRCTHWToSMT
#include "circt/Conversion/CombToSMT.h"       // createConvertCombToSMT             CIRCTCombToSMT
#include "circt/Conversion/VerifToSMT.h"      // createConvertVerifToSMT            CIRCTVerifToSMT
#include "circt/Support/Passes.h"             // createSimpleCanonicalizerPass      CIRCTSupport
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"  // exportSMTLIB                       MLIRExportSMTLIB

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteSMTImpl::run(const std::vector<std::string>& args,
                       mlir::MLIRContext &context, mlir::ModuleOp inputModule,
                       mlir::OwningOpRef<mlir::ModuleOp>& outputModule) {
    BackendImplOptions opts;
    if (!parserOptions(args, opts)) {
        log("[write_smt]: parse options failed\n\n");
        return false;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return false;
    }

    mlir::PassManager pm(&context);
    pm.addPass(circt::createConvertHWToSMT());
    pm.addPass(circt::createConvertCombToSMT());
    pm.addPass(circt::createConvertVerifToSMT());
    pm.addPass(circt::createSimpleCanonicalizerPass());

    if (failed(pm.run(inputModule))) {
        log("[write_smt]: PassManager run() failed\n\n");
        return false;
    }

    
    if (failed(mlir::smt::exportSMTLIB(inputModule, outputFile.value()->os()))) {
        log("write_smt]: exportSMTLIB failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
