#ifndef EQUIVFUSION_EQUIV_MITER_H
#define EQUIVFUSION_EQUIV_MITER_H

#include "infrastructure/utils/namespace_macro.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "circt-passes/Miter/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterToolOptions {
    std::string firstModuleName;
    std::string secondModuleName;
    std::vector<std::string> inputFilenames;
    std::string outputFilename {"-"};
    circt::EquivFusionMiter::MiterModeEnum miterMode {circt::EquivFusionMiter::MiterModeEnum::SMTLIB};
};

class EquivMiterTool {
public:
    EquivMiterTool() = default;

private:
    static bool parseOptions(const std::vector<std::string>& args, EquivMiterToolOptions& opts);

private:
    static llvm::LogicalResult miterToSMT(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                          const circt::EquivFusionMiterOptions& miterOpts);
    static llvm::LogicalResult miterToAIGER(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                            const circt::EquivFusionMiterOptions& miterOpts);
    static llvm::LogicalResult miterToBTOR2(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                            const circt::EquivFusionMiterOptions& miterOpts);

public:
    static void help(const std::string& name, const std::string& description);

    static bool run(const std::vector<std::string>& args,
                    mlir::MLIRContext &context, mlir::ModuleOp inputModule, 
                    mlir::OwningOpRef<mlir::ModuleOp>& outputModule);
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_H
