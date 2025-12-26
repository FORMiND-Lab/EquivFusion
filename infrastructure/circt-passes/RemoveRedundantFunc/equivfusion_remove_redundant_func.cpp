//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "circt-passes/RemoveRedundantFunc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"


namespace circt {

#define GEN_PASS_DEF_EQUIVFUSIONREMOVEREDUNDANTFUNCPASS
#include "circt-passes/RemoveRedundantFunc/Passes.h.inc"

} // namespace circt



namespace {

struct EquivFusionRemoveRedundantPass : public circt::impl::EquivFusionRemoveRedundantFuncPassBase<EquivFusionRemoveRedundantPass> {
    using circt::impl::EquivFusionRemoveRedundantFuncPassBase<EquivFusionRemoveRedundantPass>::EquivFusionRemoveRedundantFuncPassBase;

    void runOnOperation() override;
};

}

void EquivFusionRemoveRedundantPass::runOnOperation() {
    for (auto funcOp : llvm::make_early_inc_range(getOperation().getOps<mlir::func::FuncOp>())) {
        if (funcOp.getSymName() == this->topFunc) {
            continue;
        }
        funcOp.erase();
    }
}





