//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "circt-passes/HWTransforms/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace equivfusion {
namespace hw {
#define GEN_PASS_DEF_EQUIVFUSIONHWAGGREGATETOCOMB
#include "circt-passes/HWTransforms/Passes.h.inc"
} // namespace hw
} // namespace equivfusion
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/**
 * Convert hw.struct_create to comb.concat
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                             |       After Convert
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      %0 = hw.struct_create (%in, %in) : !hw.struct<a: i32, b: i32>       |       %0 = comb.concat %in, %in : i32, i32
 *                                                                          |       %1 = hw.bitcast %0 : (i64) -> !hw.struct<a: i32, b: i32>
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
struct HWStructCreateOpConversion : OpConversionPattern<hw::StructCreateOp> {
    using OpConversionPattern<hw::StructCreateOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, adaptor.getInput());
        return success();
    }
};


/**
 * Convert hw.struct_extract to comb.extract
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                             |       After Convert
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 *      %b = hw.struct_extract %0["b"] : !hw.struct<a: i32, b: i32>         |       %1 = hw.bitcast %0 : (!hw.struct<a: i32, b: i32>) -> i64
 *                                                                          |       %2 = comb.extract %1 from 32 : (i64) -> i32
 * -------------------------------------------------------------------------------------------------------------------------------------------------
 */
struct HWStructExtractOpConversion : OpConversionPattern<hw::StructExtractOp> {
    using OpConversionPattern<hw::StructExtractOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto structType = hw::type_cast<hw::StructType>(op.getInput().getType());
        auto fieldName = op.getFieldName();

        uint64_t offset = 0;
        for (const auto &field: structType.getElements()) {
            if (field.name == fieldName) {
                break;
            }
            offset += hw::getBitWidth(field.type);
        }

        auto fieldWidth = hw::getBitWidth(op.getResult().getType());
        auto extracted = rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), adaptor.getInput(), offset, fieldWidth);
        rewriter.replaceOp(op, extracted);
        return success();
    }
};

class AggregateTypeConverter : public TypeConverter {
public:
    AggregateTypeConverter() {
        addConversion([](Type type) -> Type { return type; });
        addConversion([](hw::StructType t) -> Type {
            return IntegerType::get(t.getContext(), hw::getBitWidth(t));
        });
        addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                    mlir::ValueRange inputs,
                                    mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1)
                return Value();

            return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
                    ->getResult(0);
        });

        addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type resultType,
                                    mlir::ValueRange inputs,
                                    mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1)
                return Value();

            return hw::BitcastOp::create(builder, loc, resultType, inputs[0])
                    ->getResult(0);
        });
    }
};

} // namespace

static void populateHWAggregateToCombOpConversionPatterns(
        RewritePatternSet &patterns, AggregateTypeConverter &typeConverter) {
    patterns.add<HWStructCreateOpConversion, HWStructExtractOpConversion>(
            typeConverter, patterns.getContext());
}

namespace {
struct EquivFusionHWAggregateToCombPass
        : public circt::equivfusion::hw::impl::EquivFusionHWAggregateToCombBase<EquivFusionHWAggregateToCombPass> {
    void runOnOperation() override;
};
} // namespace

void EquivFusionHWAggregateToCombPass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addIllegalOp<hw::StructCreateOp, hw::StructExtractOp>();
    target.addLegalDialect<hw::HWDialect, comb::CombDialect>();

    RewritePatternSet patterns(&getContext());
    AggregateTypeConverter typeConverter;
    populateHWAggregateToCombOpConversionPatterns(patterns, typeConverter);

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns))))
        return signalPassFailure();
}