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
        auto inputArrayType = cast<hw::ArrayType>(op.getInput().getType());
        auto resArrayType = cast<hw::ArrayType>(op.getType());

        /* Special hw.array_slice: need hw.array_get  [CIRCT version: firtool-1.132.0]
         * Verilog:
         *      module array(input in_2dim[1:0][1:0], input idx, output o);
         *          assign o = in_2dim[idx][idx];
         *      endmodule
         * MLIR:
         *      module {
         *          hw.module @array(in %in_2dim : !hw.array<2xarray<2xi1>>, in %idx : i1, out o : i1) {
         *              %0 = hw.array_slice %in_2dim[%idx] : (!hw.array<2xarray<2xi1>>) -> !hw.array<2xi1>
         *              %1 = hw.array_get %0[%idx] : !hw.array<2xi1>, i1
         *              hw.output %1 : i1
         *          }
         *      }
        */
        if (resArrayType == inputArrayType.getElementType()) {
            rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, op.getInput(), op.getLowIndex());
            return success();
        }

        assert(inputArrayType.getElementType() == resArrayType.getElementType());
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
