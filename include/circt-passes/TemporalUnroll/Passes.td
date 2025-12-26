//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#ifndef EQUIVFUSION_TEMPORAL_UNROLL_PASSES_TD
#define EQUIVFUSION_TEMPORAL_UNROLL_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def EquivFusionTemporalUnroll : Pass<"equivfusion-temporal-unroll", "::mlir::ModuleOp"> {
    let summary = "Temporal Unroll module";
    let description = [{
        This Pass temporal unrolls the module.
    }];

    let options = [
        Option<"timeSteps", "time-steps", "unsigned", /*default=*/"1",
                "Steps to unroll.">
    ];

    let dependentDialects = ["circt::seq::SeqDialect"];
}

#endif // EQUIVFUSION_TEMPORAL_UNROLL_PASSES_TD