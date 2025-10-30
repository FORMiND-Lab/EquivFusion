#include "libs/Backend/mlir/mlir.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteMLIRImpl::run(const std::vector<std::string>& args, mlir::MLIRContext& context,
                        mlir::ModuleOp inputModule, mlir::OwningOpRef<mlir::ModuleOp>& outputModule) {
    BackendImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("write_mlir]: parse options failed\n\n");
        return false;
    }

    if (!inputModule) {
        return true;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        llvm::errs() << errorMessage << "\n";
        return false;
    }

    inputModule.print(outputFile.value()->os());
    outputFile.value()->keep();
    
    return true;
}


XUANSONG_NAMESPACE_HEADER_END
