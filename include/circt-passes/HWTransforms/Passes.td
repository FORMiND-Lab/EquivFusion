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

#endif // EQUIVFUSION_HW_TRANSFORMS_PASSES_TD
