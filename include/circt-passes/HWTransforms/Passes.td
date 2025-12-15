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

def EquivFusionHWAggregateOpsConvert : Pass<"EquivFusion-hw-aggregate-ops-convert", "mlir::ModuleOp"> {
    let summary = "hw aggregate operations convert";

    let description = [{
        This Pass converts hw.aggregate operations

        1. hw.array_slice       => hw.array_get + hw.array_create
        2. hw.struct_explode    => hw.struct_extract
        3. hw.struct_inject     => hw.struct_extract + hw.struct_create
    }];

    let dependentDialects = ["circt::hw::HWDialect"];
}

def EquivFusionHWAggregateToComb : Pass<"EquivFusion-hw-aggregate-to-comb", "mlir::ModuleOp"> {
    let summary = "Lower aggregate operations to comb operations";

    let description = [{
        This Pass lower aggregate operations to comb operations
    }];

    let dependentDialects = ["circt::comb::CombDialect"];
}

def EquivFusionInitAfterDefinedOperands : Pass<"EquivFusion-init-after-defined-operands", "mlir::ModuleOp"> {
    let summary = "Init after defined operands in ArrayInjectOp/StructInjectOp";

    let description = [{
        This Pass init after defined operand in ArrayInjectOp and StructInjectOp
    }];

    let dependentDialects = ["circt::hw::HWDialect"];
}

#endif // EQUIVFUSION_HW_TRANSFORMS_PASSES_TD
