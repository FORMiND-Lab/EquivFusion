#include "backend/write_smt.h"

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

bool WriteSMTImpl::run(mlir::MLIRContext &context, mlir::ModuleOp module) {
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(outputFilename_, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return false;
    }

    mlir::PassManager pm(&context);
    pm.addPass(circt::createConvertHWToSMT());
    pm.addPass(circt::createConvertCombToSMT());
    pm.addPass(circt::createConvertVerifToSMT());
    pm.addPass(circt::createSimpleCanonicalizerPass());

    if (failed(pm.run(module)))
        return false;

    
    if (failed(mlir::smt::exportSMTLIB(module, outputFile.value()->os()))) {
        return false;
    }

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
