//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "circt-passes/MemrefHLS/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Matchers.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/APInt.h"

namespace circt {

#define GEN_PASS_DEF_EQUIVFUSIONCONVERTINDEXTOI32PASS
#include "circt-passes/MemrefHLS/Passes.h.inc"

} // namespace circt

namespace {

struct EquivFusionConvertIndexToI32Pass : public circt::impl::EquivFusionConvertIndexToI32PassBase<EquivFusionConvertIndexToI32Pass> {
    using circt::impl::EquivFusionConvertIndexToI32PassBase<EquivFusionConvertIndexToI32Pass>::EquivFusionConvertIndexToI32PassBase;

    void runOnOperation() override;
};

}

void EquivFusionConvertIndexToI32Pass::runOnOperation() {
    mlir::OpBuilder builder(&getContext());
    mlir::func::FuncOp funcOp = llvm::dyn_cast<mlir::func::FuncOp>(getOperation());

    funcOp.walk([&](mlir::Block *block){
        for (mlir::Value arg : block->getArguments()) {
            arg.setType(arg.getType().isIndex() ? builder.getI32Type() : arg.getType());
        }
    });

    funcOp.walk([&](mlir::Operation *op) {
        for (mlir::Value result : op->getResults()) {
            if (!result.getType().isIndex()) { 
                continue;
            }

            result.setType(result.getType().isIndex() ? builder.getI32Type() : result.getType());
            auto constant = llvm::dyn_cast<mlir::arith::ConstantOp>(op);
            if (!constant) {
                continue;
            }

            llvm::APInt value;
            mlir::detail::constant_int_value_binder(&value).match(op);

            builder.setInsertionPoint(constant);
            mlir::arith::ConstantOp newConstant = 
                    builder.create<mlir::arith::ConstantOp>(constant->getLoc(), builder.getI32IntegerAttr(value.getSExtValue()));

            result.replaceAllUsesWith(newConstant.getResult());
            constant->erase();
        }
    });
}

