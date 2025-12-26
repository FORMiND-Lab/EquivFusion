//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "circt-passes/CombTransforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace equivfusion {
namespace comb {
#define GEN_PASS_DEF_EQUIVFUSIONDECOMPOSECONCAT
#include "circt-passes/CombTransforms/Passes.h.inc"
}
}
} // namespace circt

using namespace circt;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {
static Value decomposeConcatOp(comb::ConcatOp op, OperandRange operands,
                               PatternRewriter &rewriter) {
    switch (operands.size()) {
    case 0:
        assert(0 && "cannot be called with empty operand range");
        break;
    case 1:
        return operands[0];
    case 2:
        return comb::ConcatOp::create(rewriter, op.getLoc(), operands[0], operands[1]);
    default:
        auto firstHalf = operands.size() / 2;
        auto lhs = decomposeConcatOp(op, operands.take_front(firstHalf), rewriter);
        auto rhs = decomposeConcatOp(op, operands.drop_front(firstHalf), rewriter);
        return comb::ConcatOp::create(rewriter, op.getLoc(), lhs, rhs);
    }
    
    return Value();
}

struct ConcatOpConversion : OpRewritePattern<comb::ConcatOp> {
    using OpRewritePattern<comb::ConcatOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(comb::ConcatOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.getInputs().size() <= 2)
            return failure();

        rewriter.replaceOp(op, decomposeConcatOp(op, op.getOperands(), rewriter));
        return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Decompose Concat pass
//===----------------------------------------------------------------------===//

namespace {
struct EquivFusionDecomposeConcatPass
        : public circt::equivfusion::comb::impl::EquivFusionDecomposeConcatBase<EquivFusionDecomposeConcatPass> {
    void runOnOperation() override;
};
} // namespace


void EquivFusionDecomposeConcatPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConcatOpConversion>(patterns.getContext());
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
        return signalPassFailure();
}

