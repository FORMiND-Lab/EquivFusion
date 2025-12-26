//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "circt-passes/ConvertCaseToLogicalComparsion/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/LogicalResult.h"

namespace circt {

namespace comb {

#define GEN_PASS_DEF_EQUIVFUSIONCONVERTCASETOLOGICALCOMPARSIONPASS
#include "circt-passes/ConvertCaseToLogicalComparsion/Passes.h.inc"

} // namespace comb

} // namespace circt


namespace {

struct ICmpCoversionPattern : public mlir::OpConversionPattern<circt::comb::ICmpOp> {
    using mlir::OpConversionPattern<circt::comb::ICmpOp>::OpConversionPattern;
    using OpAdaptor = typename circt::comb::ICmpOp::Adaptor;

    circt::comb::ICmpPredicate convertPredicate(circt::comb::ICmpPredicate predicate) const {
        switch (predicate) {
            case circt::comb::ICmpPredicate::ceq:
                return circt::comb::ICmpPredicate::eq;
            case circt::comb::ICmpPredicate::cne:
                return circt::comb::ICmpPredicate::ne;
            default:
                return predicate;
        }
    }


    llvm::LogicalResult matchAndRewrite(circt::comb::ICmpOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<circt::comb::ICmpOp>(op, convertPredicate(op.getPredicate()), adaptor.getLhs(), adaptor.getRhs());
        return llvm::success();
    }
};

struct EquivFusionConvertCaseToLogicalComparsionPass : \
        public circt::comb::impl::EquivFusionConvertCaseToLogicalComparsionPassBase<EquivFusionConvertCaseToLogicalComparsionPass> {
    using circt::comb::impl::EquivFusionConvertCaseToLogicalComparsionPassBase<EquivFusionConvertCaseToLogicalComparsionPass>::EquivFusionConvertCaseToLogicalComparsionPassBase;
    void runOnOperation() override;
};



} // namespace



void EquivFusionConvertCaseToLogicalComparsionPass::runOnOperation() { 
    auto *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<circt::comb::ICmpOp>([](circt::comb::ICmpOp op) {
        auto pred = op.getPredicate();
        return pred != circt::comb::ICmpPredicate::ceq && 
               pred != circt::comb::ICmpPredicate::cne;
    });

    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ICmpCoversionPattern>(patterns.getContext());

    if (llvm::failed(mlir::applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
}

