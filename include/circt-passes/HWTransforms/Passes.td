#ifndef EQUIVFUSION_HW_TRANSFORMS_PASSES_TD
#define EQUIVFUSION_HW_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionFlattenIOArray : Pass<"EquivFusion-flatten-io-array", "mlir::ModuleOp"> {
    let summary = "Convert packed-array port of hw.module to integer";

    let description = [{
        This Pass converts packed-array port of 'hw.module' to integer
    }];

    let dependentDialects = ["circt::hw::HWDialect"];
}

def EquivFusionFlattenArraySlice : Pass<"EquivFusion_flatten-array-slice", "mlir::ModuleOp"> {
    let summary = "Flatten hw.array_slice to hw.array_get + hw.array_create";

    let description = [{
        This Pass flatten hw.array_slice to hw.array_get + hw.array_create

        ```mlir
        %0 = hw.array_slice %in_1dim[%idx] : (!hw.array<3xi1>) -> !hw.array<2xi1>
        ```

        becomes:

        ```mlir
        %c1_i2 = hw.constant 1 : i2
        %c0_i2 = hw.constant 0 : i2
        %0 = comb.add %idx, %c0_i2 : i2
        %1 = hw.array_get %in_1dim[%0] : !hw.array<3xi1>, i2
        %2 = comb.add %idx, %c1_i2 : i2
        %3 = hw.array_get %in_1dim[%2] : !hw.array<3xi1>, i2
        %4 = hw.array_create %1, %3 : i1
        ```
    }];


    let dependentDialects = ["circt::hw::HWDialect"];
}

#endif // EQUIVFUSION_HW_TRANSFORMS_PASSES_TD
