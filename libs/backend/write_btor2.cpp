#include "backend/write_btor2.h"

#include "circt/Dialect/HW/HWDialect.h"

#include "circt/Dialect/HW/HWPasses.h"              // createFlattenModules()               CIRCTHWTransforms
#include "circt-passes/DecomposeConcat/Passes.h"    // createEquivFusionDecomposeConcat()    EquivFusionPassDecomposeConcat
#include "circt/Dialect/Arc/ArcPasses.h"            // createSimplifyVariadicOpsPass()       CIRCTArcTransforms
#include "circt/Conversion/HWToBTOR2.h"             // createConvertHWToBTOR2Pass()          CIRCTHWToBTOR2

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteBTOR2Impl::run(mlir::MLIRContext &context, mlir::ModuleOp module) {
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(outputFilename_, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return false;
    }
    
    mlir::PassManager pm(&context);
    pm.addPass(circt::hw::createFlattenModules());
    pm.addPass(circt::createEquivFusionDecomposeConcat());
    pm.addPass(circt::arc::createSimplifyVariadicOpsPass());

    pm.addPass(circt::createConvertHWToBTOR2Pass(outputFile.value()->os()));
    
    if (failed(pm.run(module))) {
        return false;
    }

    outputFile.value()->keep();
    return true;
}


XUANSONG_NAMESPACE_HEADER_END

