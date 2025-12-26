//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EQUIVFUSION_FUNC_TO_HW_MODULE_PASSES_TD
#define EQUIVFUSION_FUNC_TO_HW_MODULE_PASSES_TD

include "mlir/Pass/PassBase.td"

def EquivFusionFuncToHWModule : Pass<"equivfusion-func-to-hw-module", "mlir::ModuleOp"> {
    let summary = "Convert func.func to hw.module";

    let description = [{
        This Pass converts operation 'func.func' to 'hw.module'.
    }];

    let dependentDialects = [
        "mlir::func::FuncDialect", 
        "circt::hw::HWDialect", 
        "mlir::cf::ControlFlowDialect", 
        "circt::comb::CombDialect",
        "mlir::memref::MemRefDialect"
    ];
}

#endif // EQUIVFUSION_FUNC_TO_HW_MODULE_PASSES_TD

