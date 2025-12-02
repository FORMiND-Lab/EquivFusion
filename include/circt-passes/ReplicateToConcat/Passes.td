#ifndef EQUIVFUSION_REPLICATE_TO_CONCAT_PASSES_TD
#define EQUIVFUSION_REPLICATE_TO_CONCAT_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionReplicateToConcat : Pass<"EquivFusion-replicate-to-concat", "::mlir::ModuleOp"> {
    let summary = "Convert 'comb.replicate' to 'comb.concat'";

    let description = [{
        This Pass converts 'comb.replicate' to 'comb.concat'

        ```mlir
        %result = comb.replicate %a : (i1) -> i2
        ```

        becomes
        ```mlir
        %result = comb.concat %a, %a : i1, i1
        ```
    }];

    let dependentDialects = ["circt::comb::CombDialect"];
}

#endif // EQUIVFUSION_REPLICATE_TO_CONCAT_PASSES_TD
