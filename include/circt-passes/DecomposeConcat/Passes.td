#ifndef EQUIVFUSION_DECOMPOSE_CONCAT_PASSES_TD
#define EQUIVFUSION_DECOMPOSE_CONCAT_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionDecomposeConcat : Pass<"EquivFusion-decompose-concat", "::mlir::ModuleOp"> {
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


#endif // EQUIVFUSION_DECOMPOSE_CONCAT_PASSES_TD
