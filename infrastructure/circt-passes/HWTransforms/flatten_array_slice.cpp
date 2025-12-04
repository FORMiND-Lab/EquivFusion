#include "circt-passes/HWTransforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace equivfusion {
namespace hw {
#define GEN_PASS_DEF_EQUIVFUSIONFLATTENARRAYSLICE
#include "circt-passes/HWTransforms/Passes.h.inc"
} // namespece hw
} // namespace equivfusion
} // namespace circt

using namespace circt;

namespace {
struct ArraySliceOpConversion : OpRewritePattern<hw::ArraySliceOp> {
    using OpRewritePattern<hw::ArraySliceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hw::ArraySliceOp op, PatternRewriter &rewriter) const override {
        auto resArrayType = cast<hw::ArrayType>(op.getType());
        size_t sliceSize = resArrayType.getNumElements();
        if (sliceSize == 0)
            return failure();

        SmallVector<Value> elementValues;
        elementValues.reserve(sliceSize);
        auto inputArray = op.getInput();

        auto lowIndex = op.getLowIndex();
        auto indexType = lowIndex.getType();
        for (uint64_t idx = 0; idx < sliceSize; idx++) {
            Value indexValue = comb::AddOp::create(rewriter, op.getLoc(), lowIndex,
                                                   hw::ConstantOp::create(rewriter, op.getLoc(), indexType, idx));
            Value elementValue = hw::ArrayGetOp::create(rewriter, op.getLoc(), inputArray, indexValue);
            elementValues.push_back(elementValue);
        }
        rewriter.replaceOpWithNewOp<hw::ArrayCreateOp>(op, elementValues);

        return success();
    }
};
} // namespace

namespace {
struct EquivFusionFlattenArraySlicePass
        : public circt::equivfusion::hw::impl::EquivFusionFlattenArraySliceBase<EquivFusionFlattenArraySlicePass> {
    void runOnOperation() override;
};
} // namespace

void EquivFusionFlattenArraySlicePass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<ArraySliceOpConversion>(patterns.getContext());
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
        return signalPassFailure();
}
