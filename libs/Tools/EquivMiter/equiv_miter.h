#ifndef EQUIVFUSION_EQUIV_MITER_H
#define EQUIVFUSION_EQUIV_MITER_H

#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include "circt-passes/Miter/Passes.h"

XUANSONG_NAMESPACE_HEADER_START


struct EquivMiterToolOptions {
    bool printIR {false};
    circt::equivfusion::MiterModeEnum miterMode {circt::equivfusion::MiterModeEnum::SMTLIB};
    std::string specModuleName;
    std::string implModuleName;
    std::vector<std::string> inputFilenames;
    std::string outputFilename {"-"};
};

class EquivMiterTool {
public:
    EquivMiterTool() = default;

private:
    static bool parseOptions(const std::vector<std::string>& args, EquivMiterToolOptions& opts);
    static bool mergeModules(mlir::ModuleOp dest, mlir::ModuleOp src, EquivMiterToolOptions& opts, ModuleTypeEnum moduleType);

private:
    static void populatePreparePasses(mlir::PassManager& pm);

    static llvm::LogicalResult executeMiterToSMT(mlir::PassManager &pm, mlir::ModuleOp module, llvm::raw_ostream &os,
                                                 const circt::equivfusion::EquivFusionMiterOptions &miterOpts);

    static llvm::LogicalResult executeMiterToAIGER(mlir::PassManager &pm, mlir::ModuleOp module, llvm::raw_ostream &os,
                                                   const circt::equivfusion::EquivFusionMiterOptions &miterOpts);

    static llvm::LogicalResult executeMiterToBTOR2(mlir::PassManager &pm, mlir::ModuleOp module, llvm::raw_ostream &os,
                                                   const circt::equivfusion::EquivFusionMiterOptions &miterOpts);

public:
    static bool executeMiter(const std::vector<std::string>& args);
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_H
