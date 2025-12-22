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

def EquivFusionGetGlobalToAllocPass : Pass<"equivfusion-get-global-to-alloc", "mlir::func::FuncOp"> {
    let summary = "Convert 'memref.global_get' to 'memref.alloc' + 'affine.store'";

    let description = [{
        This Convert 'memref.global_get' to 'memref.alloc' + 'affine.store' in 'func.func'.
    }];

    let dependentDialects = ["mlir::func::FuncDialect", "mlir::arith::ArithDialect", "mlir::affine::AffineDialect", "mlir::memref::MemRefDialect"];
}
 
def EquivFusionFlattenMemRefPass : Pass<"equivfusion-flatten-memref", "::mlir::ModuleOp"> {
  let summary = "Flatten memrefs";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories.}];

  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def EquivFusionFlattenMemRefCallsPass : Pass<"equivfusion-flatten-memref-calls", "::mlir::ModuleOp"> {
  let summary = "Flatten memref calls";
  let description = [{
    Flattens calls to functions which have multidimensional memrefs as arguments.
    This is done by casting memref arguments through memref.subview operations.
    Any called functions which had their type signatures changes will be replaced
    by a private function definition with the new signature.
    It is up to users of this pass to define how these rewritten functions are
    to be implemented.}];

  let dependentDialects = ["mlir::memref::MemRefDialect"];
}


#endif // EQUIVFUSION_MEMREF_HLS_PASSES_TD