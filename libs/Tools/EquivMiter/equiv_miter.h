#ifndef EQUIVFUSION_EQUIV_MITER_H
#define EQUIVFUSION_EQUIV_MITER_H

#include "infrastructure/utils/namespace_macro.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "circt-passes/Miter/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterImplOptions {
    std::string firstModuleName;
    std::string secondModuleName;
    std::vector<std::string> inputFilenames;
    std::string outputFilename;
    circt::EquivFusionMiter::MiterModeEnum miterMode {circt::EquivFusionMiter::MiterModeEnum::SMTLIB};
};

class EquivMiterImpl {
public:
    EquivMiterImpl() = default;

private:
    static bool parserOptions(const std::vector<std::string>& args, EquivMiterImplOptions& opts);

public:
    static void help(const std::string& name, const std::string& description);

    static bool run(const std::vector<std::string>& args,
                    mlir::MLIRContext &context, mlir::ModuleOp inputModule, 
                    mlir::OwningOpRef<mlir::ModuleOp>& outputModule);
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_H
