#ifndef EQUIVFUSION_FUNC_TO_HW_MODULE_PASSES_TD
#define EQUIVFUSION_FUNC_TO_HW_MODULE_PASSES_TD

include "mlir/Pass/PassBase.td"

def FuncToHWModule : Pass<"func-to-hw-module", "mlir::ModuleOp"> { 
    let summary = "Convert func.func to hw.module";

    let description = [{
        This Pass converts operation 'func.func' to 'hw.module'.
    }];

    let dependentDialects = [
        "mlir::func::FuncDialect", 
        "circt::hw::HWDialect", 
        "mlir::cf::ControlFlowDialect", 
        "circt::comb::CombDialect",
        "mlir::memref::MemRefDialect"
    ];
}

#endif // EQUIVFUSION_FUNC_TO_HW_MODULE_PASSES_TD

