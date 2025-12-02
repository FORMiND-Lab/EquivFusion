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

enum struct DesignTypeEnum {
    SPEC,
    IMPL,
};

struct EquivMiterToolOptions {
    bool printIR {false};
    circt::EquivFusionMiter::MiterModeEnum miterMode {circt::EquivFusionMiter::MiterModeEnum::SMTLIB};
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
    static bool mergeModules(mlir::ModuleOp dest, mlir::ModuleOp src, EquivMiterToolOptions& opts, DesignTypeEnum designType);

private:
    static void populatePreparePasses(mlir::PassManager& pm);

    static llvm::LogicalResult miterToSMT(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                          const circt::EquivFusionMiterOptions& miterOpts);
    static llvm::LogicalResult miterToAIGER(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                            const circt::EquivFusionMiterOptions& miterOpts);
    static llvm::LogicalResult miterToBTOR2(mlir::PassManager& pm, mlir::ModuleOp module, llvm::raw_ostream& os,
                                            const circt::EquivFusionMiterOptions& miterOpts);

public:
    static void help(const std::string& name, const std::string& description);

    static bool run(const std::vector<std::string>& args);
};

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_H
