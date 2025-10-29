#include "infrastructure/log/log.h"
#include "libs/Frontend/read_mlir.h"

#include "mlir/Parser/Parser.h"     // parseSourceFile  MLIRParser

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool ReadMLIRImpl::run(const std::vector<std::string>& args,
                       mlir::MLIRContext &context,
                       mlir::ModuleOp module,
                       mlir::OwningOpRef<mlir::ModuleOp>& outputModule) {
    FrontendImplOptions opts;
    if (!initOptions(args, opts)) {
        log("read_mlir: options error\n\n");
        return false;
    }    

    if (!module) {
        module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    }
 
    for (const auto& inputFilename : opts.inputFilenames) {
        auto fileModule = parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
        if (fileModule) {
            mergeModules(module, fileModule.get());
        }
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

    outputModule = std::move(module);
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
