t
#include "circt-passes/HWTransforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace equivfusion {
namespace hw {
#define GEN_PASS_DEF_EQUIVFUSIONHWAGGREGATEOPSCONVERT
#include "circt-passes/HWTransforms/Passes.h.inc"
} // namespece hw
} // namespace equivfusion
} // namespace circt

using namespace circt;

namespace {

/**
 * Convert hw.array_slice to hw.array_get + hw.array_create
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *    Example                                                                      |       After Convert
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *    %0 = hw.array_slice %in_1dim[%idx] : (!hw.array<3xi1>) -> !hw.array<2xi1>    |       %c1_i2 = hw.constant 1 : i2
 *                                                                                 |       %c0_i2 = hw.constant 0 : i2
 *                                                                                 |       %0 = comb.add %idx, %c0_i2 : i2
 *                                                                                 |       %1 = hw.array_get %in_1dim[%0] : !hw.array<3xi1>, i2
 *                                                                                 |       %2 = comb.add %idx, %c1_i2 : i2
 *                                                                                 |       %3 = hw.array_get %in_1dim[%2] : !hw.array<3xi1>, i2
 *                                                                                 |       %4 = hw.array_create %1, %3 : i1
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
struct HWArraySliceOpConversion : OpRewritePattern<hw::ArraySliceOp> {
    using OpRewritePattern<hw::ArraySliceOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hw::ArraySliceOp op, PatternRewriter &rewriter) const override {
        auto inputArrayType = cast<hw::ArrayType>(op.getInput().getType());
        auto resArrayType = cast<hw::ArrayType>(op.getType());

        /**
         * Special hw.array_slice: need hw.array_get  [CIRCT version: firtool-1.132.0]
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

/**
 * Convert hw.struct_explode to hw.struct_extract
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                             |       After Convert
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      %a, %b = hw.struct_explode %0 : !hw.struct<a: i32, b: i32>          |       %a = hw.struct_extract %0["a"] : !hw.struct<a: i32, b: i32>
 *                                                                          |       %b = hw.struct_extract %0["b"] : !hw.struct<a: i32, b: i32>
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
struct HWStructExplodeOpConversion : OpRewritePattern<hw::StructExplodeOp> {
    using OpRewritePattern<hw::StructExplodeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hw::StructExplodeOp op, PatternRewriter &rewriter) const override {
        auto opResults = op.getResults();
        auto elements = type_cast<hw::StructType>(op.getInput().getType()).getElements();
        assert(opResults.size() == elements.size());

        SmallVector<Value> newResults;
        newResults.reserve(opResults.size());
        for (uint32_t index = 0; index < elements.size(); index++) {
            auto extractOp = hw::StructExtractOp::create(rewriter, op.getLoc(), op.getInput(), elements[index].name);
            rewriter.replaceAllUsesWith(opResults[index], extractOp);
        }
        return success();
    }
};


/**
 * Convert hw.struct_inject to hw.struct_extract + hw.struct_create
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                             |       After Convert
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      %1 = hw.struct_inject %0["a"], %in : !hw.struct<a: i32, b: i32>     |       %b = hw.struct_extract %0["b"] : !hw.struct<a: i32, b: i32>
 *                                                                          |       %1 = hw.struct_create (%in, %b) : !hw.struct<a: i32, b: i32>
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
struct HWStructInjectOpConversion : OpRewritePattern<hw::StructInjectOp> {
    using OpRewritePattern<hw::StructInjectOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(hw::StructInjectOp op, PatternRewriter &rewriter) const override {
        auto structType = cast<hw::StructType>(op.getInput().getType());
        auto fieldName = op.getFieldName();
        SmallVector<Value> fieldValues;
        for (const auto &field : structType.getElements()) {
            Value fieldValue = (field.name == fieldName) ? op.getNewValue() :
                               hw::StructExtractOp::create(rewriter, op.getLoc(), op.getInput(), field.name);
            fieldValues.push_back(fieldValue);
        }

        rewriter.replaceOpWithNewOp<hw::StructCreateOp>(op, structType, fieldValues);
        return success();
    }
};

} // namespace

namespace {
struct EquivFusionHWAggregateOpsConvertPass
        : public circt::equivfusion::hw::impl::EquivFusionHWAggregateOpsConvertBase<EquivFusionHWAggregateOpsConvertPass> {
    void runOnOperation() override;
};
} // namespace

void EquivFusionHWAggregateOpsConvertPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    patterns.add<HWArraySliceOpConversion,
                 HWStructExplodeOpConversion,
                 HWStructInjectOpConversion>(&getContext());

    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
}