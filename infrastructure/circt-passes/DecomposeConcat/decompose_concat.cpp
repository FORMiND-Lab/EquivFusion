#include "circt-passes/DecomposeConcat/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
#define GEN_PASS_DEF_EQUIVFUSIONDECOMPOSECONCAT
#include "circt-passes/DecomposeConcat/Passes.h.inc"
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
        return rewriter.create<comb::ConcatOp>(op.getLoc(), operands[0], operands[1]);
    default:
        auto firstHalf = operands.size() / 2;
        auto lhs = decomposeConcatOp(op, operands.take_front(firstHalf), rewriter);
        auto rhs = decomposeConcatOp(op, operands.drop_front(firstHalf), rewriter);
        return rewriter.create<comb::ConcatOp>(op.getLoc(), lhs, rhs);
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

static void populateDecomposeConcatPatterns(RewritePatternSet &patterns) {
  patterns.add<ConcatOpConversion>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Decompose Concat pass
//===----------------------------------------------------------------------===//

namespace {
struct EquivFusionDecomposeConcatPass
    : public circt::impl::EquivFusionDecomposeConcatBase<EquivFusionDecomposeConcatPass> {
  void runOnOperation() override;
};
} // namespace


void EquivFusionDecomposeConcatPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateDecomposeConcatPatterns(patterns);
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (failed(mlir::applyPatternsGreedily(getOperation(), frozen)))
        return signalPassFailure();
}

