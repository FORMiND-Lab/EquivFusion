#include "libs/Write/btor2/btor2.h"

#include "circt/Conversion/HWToBTOR2.h"             // createConvertHWToBTOR2Pass()          CIRCTHWToBTOR2

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteBTOR2Impl::run(const std::vector<std::string>& args, mlir::MLIRContext &context, mlir::ModuleOp inputModule) {
    if (!inputModule) {
        return true;
    }

    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_btor2]: parser options failed\n\n");
        return false;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_btor2]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }
    
    mlir::PassManager pm(&context);
    pm.addPass(circt::createConvertHWToBTOR2Pass(outputFile.value()->os()));
    if (failed(pm.run(inputModule))) {
        log("[write_btor2]: run PassManager failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}


XUANSONG_NAMESPACE_HEADER_END

