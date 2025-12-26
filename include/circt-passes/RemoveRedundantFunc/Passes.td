//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EQUIVFUSION_REMOVE_REDUNDANT_FUNC_TD
#define EQUIVFUSION_REMOVE_REDUNDANT_FUNC_TD

include "mlir/Pass/PassBase.td"

def EquivFusionRemoveRedundantFuncPass : Pass<"equivfusion-remove-redundant-func", "mlir::ModuleOp"> {
    let summary = "Remove Redundant Func in 'builtin.module'";

    let description = [{
        This Pass Removes func.func except top func in 'builtin.module'.
    }];

    let dependentDialects = ["mlir::func::FuncDialect"];

    let options = [
        Option<"topFunc", "equivfusion-remove-redundant-func-top-func", 
               "std::string", /*default*/"", "Name of top func.">
    ];
}


#endif // EQUIVFUSION_REMOVE_REDUNDANT_FUNC_TD
