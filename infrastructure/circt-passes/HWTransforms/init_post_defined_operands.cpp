//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-passes/HWTransforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace equivfusion {
namespace hw {
#define GEN_PASS_DEF_EQUIVFUSIONINITPOSTDEFINEDOPERANDS
#include "circt-passes/HWTransforms/Passes.h.inc"
} // namespece hw
} // namespace equivfusion
} // namespace circt

using namespace circt;
using namespace mlir;

static bool operandNeedsInitialization(Operation* op, Value operand) {
    if (!operand.getDefiningOp()) {
        return false;
    }

    auto *defOp = operand.getDefiningOp();
    return !defOp->isBeforeInBlock(op);
}

namespace {

/**
 * Add initial value 0 for hw.array_inject/hw.struct_inject use input defined after
 *--------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                             |       After Convert
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 *      %0 = hw.array_inject %0[%c1_i2], %in : !hw.array<3xi1>, i2          |       %c0_i3 = hw.constant 0 : i3
 *                                                                          |       %0 = hw.bitcast %c0_i3 : (i3) -> !hw.array<3xi1>
 *                                                                          |       %1 = hw.array_inject %0[%c1_i2], %in : !hw.array<3xi1>, i2
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 *      %0 = hw.struct_inject %0["a"], %in : !hw.struct<a: i32, b: i32>     |       %c0_i64 = hw.constant 0 : i64
 *                                                                          |       %0 = hw.bitcast %c0_i64 : (i64) -> !hw.struct<a: i32, b: i32>
 *                                                                          |       %1 = hw.struct_inject %0["a"], %in : !hw.struct<a: i32, b: i32>
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
template <typename OpTy>
struct AggregateInjectOpConversion : OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
        auto input = op.getInput();
        if (!operandNeedsInitialization(op, input))
            return failure();

        Type inputType = input.getType();
        Value zeroValue = hw::ConstantOp::create(rewriter, op.getLoc(), APInt(hw::getBitWidth(inputType), 0));
        Value defaultValue = inputType.isInteger() ? zeroValue :
                             hw::BitcastOp::create(rewriter, op.getLoc(), inputType, zeroValue);
        rewriter.modifyOpInPlace(op, [&]() { op.getInputMutable().assign(defaultValue); });
        return success();
    }
};

} // namespace

namespace {
struct EquivFusionInitPostDefinedOperandsPass
        : public circt::equivfusion::hw::impl::EquivFusionInitPostDefinedOperandsBase<EquivFusionInitPostDefinedOperandsPass> {
    void runOnOperation() override;
};
} // namespace

void EquivFusionInitPostDefinedOperandsPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());

    // Add patterns
    patterns.add<AggregateInjectOpConversion<hw::ArrayInjectOp>,
                 AggregateInjectOpConversion<hw::StructInjectOp>>(&getContext());

    // Apply the conversion
    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
}