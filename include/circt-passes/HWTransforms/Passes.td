//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EQUIVFUSION_HW_TRANSFORMS_PASSES_TD
#define EQUIVFUSION_HW_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionFlattenIOArray : Pass<"equivfusion-flatten-io-array", "mlir::ModuleOp"> {
    let summary = "Convert packed-array port of hw.module to integer";

    let description = [{
        This Pass converts packed-array port of 'hw.module' to integer
    }];

    let dependentDialects = ["circt::hw::HWDialect"];
}

def EquivFusionHWAggregateOpsConvert : Pass<"equivfusion-hw-aggregate-ops-convert", "mlir::ModuleOp"> {
    let summary = "hw aggregate operations convert";

    let description = [{
        This Pass converts hw.aggregate operations

        1. hw.array_slice       => N x hw.array_get + hw.array_create
        2. hw.struct_explode    => N x hw.struct_extract
        3. hw.struct_inject     => N x hw.struct_extract + hw.struct_create
        4. comb.mux             => 2 x hw.bitcast + comb.mux + hw.bitcast [result type is hw.array or hw.struct]
    }];

    let dependentDialects = ["circt::hw::HWDialect"];
}

def EquivFusionHWAggregateToComb : Pass<"equivfusion-hw-aggregate-to-comb", "mlir::ModuleOp"> {
    let summary = "Lower aggregate operations to comb operations";

    let description = [{
        This Pass lower aggregate operations to comb operations
    }];

    let dependentDialects = ["circt::comb::CombDialect"];
}

def EquivFusionInitPostDefinedOperands : Pass<"equivfusion-init-post-defined-operands", "mlir::ModuleOp"> {
    let summary = "Init post defined operands in ArrayInjectOp/StructInjectOp";

    let description = [{
        This Pass init post defined operand in ArrayInjectOp and StructInjectOp
    }];

    let dependentDialects = ["circt::hw::HWDialect"];
}

#endif // EQUIVFUSION_HW_TRANSFORMS_PASSES_TD
