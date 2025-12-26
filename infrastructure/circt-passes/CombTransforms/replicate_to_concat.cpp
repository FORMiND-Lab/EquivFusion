//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-passes/CombTransforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


namespace circt {
namespace equivfusion {
namespace comb {
#define GEN_PASS_DEF_EQUIVFUSIONREPLICATETOCONCAT
#include "circt-passes/CombTransforms/Passes.h.inc"
}
}
} // namespace circt

using namespace circt;

namespace {
struct ReplicateOpConversion : OpRewritePattern<comb::ReplicateOp> {
    using OpRewritePattern<comb::ReplicateOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(comb::ReplicateOp op,
                                  PatternRewriter &rewriter) const override {
        size_t multiple = op.getMultiple();
        SmallVector<Value> inputs(multiple, op.getInput());
        rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, inputs);

        return success();
    }
};

} //namespace

//===----------------------------------------------------------------------===//
// ReplicateOp to ConcatOp pass
//===----------------------------------------------------------------------===//

namespace {
struct EquivFusionReplicateToConcatPass
        : public circt::equivfusion::comb::impl::EquivFusionReplicateToConcatBase<EquivFusionReplicateToConcatPass> {
    void runOnOperation() override;
};
} // namespace

void EquivFusionReplicateToConcatPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplicateOpConversion>(patterns.getContext());
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
        return signalPassFailure();
}