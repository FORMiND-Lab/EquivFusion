//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-passes/HWTransforms/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace hw;

namespace circt {
namespace equivfusion {
namespace hw {
#define GEN_PASS_DEF_EQUIVFUSIONFLATTENIOARRAY
#include "circt-passes/HWTransforms/Passes.h.inc"
}
}
} // namespace circt

static bool isArrayType(Type type) {
    return isa<hw::ArrayType>(hw::getCanonicalType(type));
}

static bool isLegalHWModuleOp(hw::HWModuleOp moduleOp) {
    return llvm::none_of(moduleOp.getHWModuleType().getPortTypes(),isArrayType);
}

namespace {
class ArrayTypeConverter : public TypeConverter {
public:
    ArrayTypeConverter() {
        addConversion([](Type type) -> Type { return type; });
        addConversion([](hw::ArrayType t) -> Type {
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

struct OutputOpConversion : public OpConversionPattern<hw::OutputOp> {
    using OpConversionPattern<hw::OutputOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(hw::OutputOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<hw::OutputOp>(op, adaptor.getOperands());
        return success();
    }
};

} // namespace

namespace {
struct EquivFusionFlattenIOArrayPass
        : public circt::equivfusion::hw::impl::EquivFusionFlattenIOArrayBase<EquivFusionFlattenIOArrayPass> {
    void runOnOperation() override;
};
} // namespace

/**
 * Flatten IO array to IO bit
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 *      Example                                                                         |       After Convert
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 *      hw.module @array(in %in : !hw.array<2xi1>, out out : !hw.array<2xi1>) {         |       hw.module @array(in %in : i2, out out : i2) {
 *          hw.output %in : !hw.array<2xi1>                                             |           hw.output %in : i2
 *      }                                                                               |       }
 * --------------------------------------------------------------------------------------------------------------------------------------------------
 */
void EquivFusionFlattenIOArrayPass::runOnOperation() {
    ModuleOp module = getOperation();
    ArrayTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // Add patterns
    patterns.add<OutputOpConversion>(typeConverter, &getContext());
    hw::populateHWModuleLikeTypeConversionPattern(hw::HWModuleOp::getOperationName(), patterns, typeConverter);
    target.addDynamicallyLegalOp<hw::HWModuleOp>([](hw::HWModuleOp op) {
        return isLegalHWModuleOp(op);
    });
    target.addDynamicallyLegalOp<hw::OutputOp>([&](hw::OutputOp op) {
        return typeConverter.isLegal(op.getOperandTypes());
    });

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        return signalPassFailure();
}