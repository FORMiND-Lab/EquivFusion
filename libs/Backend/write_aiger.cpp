#include "libs/Backend/write_aiger.h"

#include "circt/Dialect/HW/HWOps.h"

#include "circt/Dialect/HW/HWPasses.h"      // createFlattenModules             CIRCTHWTransforms
#include "circt/Support/Passes.h"           // createSimpleCanonicalizerPass    CIRCTSupport
#include "circt/Conversion/CombToAIG.h"     // createConvertCombToAIG           CIRCTCombToAIG
#include "circt/Dialect/AIG/AIGPasses.h"    // createLowerVariadic              CIRCTAIGTransforms
#include "circt/Conversion/ExportAIGER.h"   // exportAIGER                      CIRCTExportAIGER

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteAIGERImpl::run(const std::vector<std::string>& args,
                         mlir::MLIRContext &context, mlir::ModuleOp inputModule,
                         mlir::OwningOpRef<mlir::ModuleOp>& outputModule) {
    BackendImplOptions opts;
    if (!parserOptions(args, opts)) {
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
    pm.addPass(circt::hw::createFlattenModules());
    pm.addPass(circt::createSimpleCanonicalizerPass());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::createConvertCombToAIG());
    pm.nest<circt::hw::HWModuleOp>().addPass(circt::aig::createLowerVariadic());

    if (failed(pm.run(inputModule)))
        return false;
    
    auto ops = inputModule.getOps<circt::hw::HWModuleOp>();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
        return false;

    if (failed(circt::aiger::exportAIGER(*ops.begin(), outputFile.value()->os())))
        return false;

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
