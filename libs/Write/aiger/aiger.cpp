#include "libs/Write/aiger/aiger.h"

#include "circt/Dialect/HW/HWOps.h"

#include "circt/Conversion/ExportAIGER.h"   // exportAIGER                      CIRCTExportAIGER

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteAIGERImpl::run(const std::vector<std::string>& args, mlir::MLIRContext &context, mlir::ModuleOp inputModule) {
    if (!inputModule) {
        return true;
    }    

    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_aiger]: parser options failed\n\n");
        return false;
    }

    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_aiger]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    auto ops = inputModule.getOps<circt::hw::HWModuleOp>();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
        return false;

    if (failed(circt::aiger::exportAIGER(*ops.begin(), outputFile.value()->os()))) {
        log("[write_aiger]: run exportAIGER failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
