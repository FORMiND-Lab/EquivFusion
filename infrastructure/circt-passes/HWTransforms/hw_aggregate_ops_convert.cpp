//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

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
        auto inputArrayType = hw::type_cast<hw::ArrayType>(op.getInput().getType());
        auto resArrayType = hw::type_cast<hw::ArrayType>(op.getType());

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
        auto elements = hw::type_cast<hw::StructType>(op.getInput().getType()).getElements();
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
        auto structType = hw::type_cast<hw::StructType>(op.getInput().getType());
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

/**
 * Convert comb.mux with hw::ArrayType or hw::StructType to hw::BitcastOp + comb.mux + hw::BitcastOp
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                             |       After Convert
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      %0 = comb.mux %cond, %in1, %in2 : !hw.array<2xi1>                   |       %0 = hw.bitcast %in1 : (!hw.array<2xi1>) -> i2
 *                                                                          |       %1 = hw.bitcast %in2 : (!hw.array<2xi1>) -> i2
 *                                                                          |       %2 = comb.mux %cond, %0, %1 : i2
 *                                                                          |       %3 = hw.bitcast %2 : (i2) -> !hw.array<2xi1>
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      %0 = comb.mux %cond, %in1, %in2 : !hw.struct<a: i32, b: i32>        |       %0 = hw.bitcast %in1 : (!hw.struct<a: i32, b: i32>) -> i64
 *                                                                          |       %1 = hw.bitcast %in2 : (!hw.struct<a: i32, b: i32>) -> i64
 *                                                                          |       %2 = comb.mux %cond, %0, %1 : i64
 *                                                                          |       %3 = hw.bitcast %2 : (i64) -> !hw.struct<a: i32, b: i32>
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
struct CombMuxOpConversion : OpRewritePattern<comb::MuxOp> {
    using OpRewritePattern<comb::MuxOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(comb::MuxOp op, PatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType();
        if (!isa<hw::StructType>(resultType) && !isa<hw::ArrayType>(resultType)) {
            return failure();
        }

        auto trueValue = op.getTrueValue();
        auto falseValue = op.getFalseValue();
        assert (trueValue.getType() == resultType && falseValue.getType() == resultType);

        auto integerType = IntegerType::get(resultType.getContext(), hw::getBitWidth(resultType));

        auto newTrueValue = hw::BitcastOp::create(rewriter, trueValue.getLoc(), integerType, trueValue);
        auto newFalseValue = hw::BitcastOp::create(rewriter, falseValue.getLoc(), integerType, falseValue);
        auto newMuxOp = comb::MuxOp::create(rewriter, op.getLoc(), op.getCond(), newTrueValue, newFalseValue);
        auto newResult = hw::BitcastOp::create(rewriter, op.getLoc(), resultType, newMuxOp);
        rewriter.replaceOp(op, newResult);
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
                 HWStructInjectOpConversion,
                 CombMuxOpConversion>(&getContext());

    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
}