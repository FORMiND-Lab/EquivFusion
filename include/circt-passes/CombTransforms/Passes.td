#ifndef EQUIVFUSION_COMB_TRANSFORMS_PASSES_TD
#define EQUIVFUSION_COMB_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionDecomposeConcat : Pass<"equivfusion-decompose-concat", "::mlir::ModuleOp"> {
  let summary = "Decompose multi-input concat operations into binary concat chains";
  let description = [{
    This pass decomposes comb.concat operations with more than 2 inputs into
    a chain of binary comb.concat operations. For example:
    
    ```mlir
    %result = comb.concat %a, %b, %c, %d : i1, i1, i1, i1 -> i4
    ```
    
    becomes:
    
    ```mlir
    %temp1 = comb.concat %a, %b : i1, i1 -> i2
    %temp2 = comb.concat %temp1, %c : i2, i1 -> i3  
    %result = comb.concat %temp2, %d : i3, i1 -> i4
    ```
  }];
  
  let dependentDialects = ["circt::comb::CombDialect"];
}

def EquivFusionReplicateToConcat : Pass<"equivfusion-replicate-to-concat", "::mlir::ModuleOp"> {
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

#endif // EQUIVFUSION_COMB_TRANSFORMS_PASSES_TD
