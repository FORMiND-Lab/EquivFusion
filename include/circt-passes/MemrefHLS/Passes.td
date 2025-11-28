#ifndef EQUIVFUSION_MEMREF_HLS_PASSES_TD
#define EQUIVFUSION_MEMREF_HLS_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionConvertIndexToI32Pass : Pass<"equivfusion-convert-index-to-i32", "mlir::func::FuncOp"> {
    let summary = "Convert all index typed values to an i32 integer.";

    let description = [{
        This pass converts all index typed values to an i32 integer.
    }];

    let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}

def EquivFusionSetFuncArgDirectionPass : Pass<"equivfusion-set-func-arg-direction", "mlir::func::FuncOp"> {
    let summary = "Set direction of arguments for func.func.";

    let description = [{
        This pass sets direction of arguments for func.func.
    }];

    let dependentDialects = ["mlir::func::FuncDialect"];

    let options = [
        ListOption<"inputPorts", "input-ports", "std::string", 
                   "List of input port names">,
        ListOption<"outputPorts", "output-ports", "std::string",
                   "List of output port names">
    ];
}




#endif // EQUIVFUSION_MEMREF_HLS_PASSES_TD