#ifndef EQUIVFUSION_EQUIV_MITER_H
#define EQUIVFUSION_EQUIV_MITER_H

#include "infrastructure/utils/namespace_macro.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "circt-passes/Miter/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

class EquivMiterTool {
public:
    EquivMiterTool() = default;

public:
    bool initOptions(const std::vector<std::string>& args);
    bool run(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp>& outputModule);

private:
    mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> parseAndMergeModules(mlir::MLIRContext &context);
    mlir::FailureOr<mlir::StringAttr> mergeModules(mlir::ModuleOp dest, mlir::ModuleOp src, mlir::StringAttr name);

private:
    std::string firstModuleName_;
    std::string secondModuleName_;
    std::vector<std::string> inputFilenames_;
    circt::EquivFusionMiter::MiterModeEnum miterMode_ {circt::EquivFusionMiter::MiterModeEnum::SMTLIB};
    bool verbose_ {false};
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_H
