#include "circt-passes/FlattenIOArray/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_EQUIVFUSIONFLATTENIOARRAY
#include "circt-passes/FlattenIOArray/Passes.h.inc"
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
    : public circt::impl::EquivFusionFlattenIOArrayBase<EquivFusionFlattenIOArrayPass> {
    using circt::impl::EquivFusionFlattenIOArrayBase<EquivFusionFlattenIOArrayPass>::EquivFusionFlattenIOArrayBase;
    void runOnOperation() override;
};
} // namespace

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