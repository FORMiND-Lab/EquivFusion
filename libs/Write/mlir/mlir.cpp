#include "libs/Write/mlir/mlir.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteMLIRImpl::run(const std::vector<std::string>& args, mlir::MLIRContext& context, mlir::ModuleOp inputModule) {
    // inputModule is empty
    if (!inputModule) {
        return true;
    }

    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_mlir]: parse options failed\n\n");
        return false;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_mlir]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    inputModule.print(outputFile.value()->os());
    outputFile.value()->keep();
    
    return true;
}


XUANSONG_NAMESPACE_HEADER_END
