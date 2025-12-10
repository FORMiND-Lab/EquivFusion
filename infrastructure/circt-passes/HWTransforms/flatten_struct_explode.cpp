#include "circt-passes/HWTransforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace equivfusion {
namespace hw {
#define GEN_PASS_DEF_EQUIVFUSIONFLATTENSTRUCTEXPLODE
#include "circt-passes/HWTransforms/Passes.h.inc"
} // namespece hw
} // namespace equivfusion
} // namespace circt

using namespace circt;

namespace {
struct StructExplodeOpConversion : OpRewritePattern<hw::StructExplodeOp> {
    using OpRewritePattern<hw::StructExplodeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hw::StructExplodeOp op, PatternRewriter &rewriter) const override {
        auto opResults = op.getResults();
        auto elements = type_cast<hw::StructType>(op.getInput().getType()).getElements();
        assert(opResults.size() == elements.size());

        SmallVector<Value> newResults;
        newResults.reserve(opResults.size());
        for (uint32_t index = 0; index < elements.size(); index++) {
            auto extractOp = hw::StructExtractOp::create(rewriter,
                                                         op.getLoc(), op.getInput(), elements[index].name);
            rewriter.replaceAllUsesWith(opResults[index], extractOp);
        }
        return success();
    }
};
} // namespace

namespace {
struct EquivFusionFlattenStructExplodePass
        : public circt::equivfusion::hw::impl::EquivFusionFlattenStructExplodeBase<EquivFusionFlattenStructExplodePass> {
    void runOnOperation() override;
};
} // namespace

void EquivFusionFlattenStructExplodePass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<StructExplodeOpConversion>(patterns.getContext());
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
        return signalPassFailure();
}
