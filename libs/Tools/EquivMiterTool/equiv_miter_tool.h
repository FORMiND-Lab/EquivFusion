#ifndef EQUIVFUSION_EQUIV_MITER_TOOL_H
#define EQUIVFUSION_EQUIV_MITER_TOOL_H

#include "infrastructure/utils/namespace_macro.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "circt-passes/Miter/Passes.h"

using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterOptions {
    std::string firstModuleName;
    std::string secondModuleName;
    std::vector<std::string> inputFilenames;
    std::string outputFilename {"-"};
    EquivFusionMiter::MiterModeEnum miterMode {EquivFusionMiter::MiterModeEnum::SMTLIB};

    bool verbose {false};
};

class EquivMiterTool {
public:
    EquivMiterTool() = default;

public:
    bool initOptions(const std::vector<std::string>& args);
    bool run();

private:
    FailureOr<OwningOpRef<ModuleOp>> parseAndMergeModules(MLIRContext &context);
    FailureOr<StringAttr> mergeModules(ModuleOp dest, ModuleOp src, StringAttr name);

private:
    LogicalResult executeMiter(MLIRContext &context, OwningOpRef<ModuleOp>& module);
    LogicalResult executeMiterToSMTLIB(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os);
    LogicalResult executeMiterToAIGER(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os);
    LogicalResult executeMiterToBTOR2(mlir::PassManager &pm, ModuleOp module, llvm::raw_ostream &os);

private:
    EquivMiterOptions options_;
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_TOOL_H
