//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-passes/MemrefHLS/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/APInt.h"


namespace circt {

#define GEN_PASS_DEF_EQUIVFUSIONGETGLOBALTOALLOCPASS
#include "circt-passes/MemrefHLS/Passes.h.inc"

} // namespace circt

namespace {

struct GetGlobalToAllocRewriter : public mlir::OpRewritePattern<mlir::memref::GetGlobalOp> {
    using mlir::OpRewritePattern<mlir::memref::GetGlobalOp>::OpRewritePattern;
    
    mlir::LogicalResult 
    assignValueToMemref(mlir::PatternRewriter &rewriter, mlir::Value memref, size_t currentDim,
        mlir::Attribute initialValueAttr, llvm::ArrayRef<int64_t> &shape, llvm::SmallVector<int64_t> &indices) const;

    mlir::LogicalResult 
    getDenseElementsAttrValue(mlir::PatternRewriter &rewriter, mlir::DenseElementsAttr denseAttr, 
        mlir::Type elementType, size_t flatIndex, mlir::Value &constVal) const;

    mlir::LogicalResult 
    getDenseResourceElementsAttrValue(mlir::PatternRewriter &rewriter, mlir::DenseResourceElementsAttr denseResourceAttr, 
        mlir::Type elementType, size_t flatIndex, mlir::Value &constVal) const;



    mlir::LogicalResult matchAndRewrite(mlir::memref::GetGlobalOp op, mlir::PatternRewriter &rewriter) const override {
        // Create 'memref.alloc' based on the result type of 'memref.get_global'.
        mlir::MemRefType memrefType = llvm::dyn_cast<mlir::MemRefType>(op.getResult().getType());
        mlir::Type elementType = memrefType.getElementType();

        if (!memrefType.hasStaticShape()) {
            llvm::errs() << "Error: Type of global memref is not a static shape!\n";
            return mlir::failure();
        }

        if (!elementType.isInteger() && !elementType.isFloat()) {
            llvm::errs() << "Error: Element type of global memerf is not integer or float!\n";
            return mlir::failure();
        }

        mlir::memref::AllocOp allocOp = rewriter.create<mlir::memref::AllocOp>(op.getLoc(), memrefType);

        // Assign values to the memref allocated by 'memref.alloc' based on global memref.
        llvm::ArrayRef<int64_t> shape = memrefType.getShape();
        mlir::Value newMemref = allocOp.getResult();

        llvm::StringRef globalMemrefName = op.getName();
        mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
        mlir::memref::GlobalOp globalOp = moduleOp.lookupSymbol<mlir::memref::GlobalOp>(globalMemrefName);

        if (!globalOp) {
            llvm::errs() << "Error: Global memref '" << globalMemrefName << "' not found!\n";
            return mlir::failure();
        }

        if (!globalOp.getConstant()) {
            llvm::errs() << "Error: Global memref '" << globalMemrefName << "' is not a constant!\n";
            return mlir::failure();
        }

        std::optional<mlir::Attribute> initValueAttr = globalOp.getInitialValue();

        if (initValueAttr.has_value() && !llvm::isa<mlir::UnitAttr>(*initValueAttr)) {
            llvm::SmallVector<int64_t> indices;
            if (mlir::failed(assignValueToMemref(rewriter, newMemref, 0, initValueAttr.value(), shape, indices))) {
                return mlir::failure();
            }
        }

        rewriter.replaceOp(op, allocOp);

        return mlir::success();
    }
};

struct EquivFusionGetGlobalToAllocPass : public circt::impl::EquivFusionGetGlobalToAllocPassBase<EquivFusionGetGlobalToAllocPass> {
    using circt::impl::EquivFusionGetGlobalToAllocPassBase<EquivFusionGetGlobalToAllocPass>::EquivFusionGetGlobalToAllocPassBase;
    
    void runOnOperation() override;
};

} // namespace

mlir::LogicalResult 
GetGlobalToAllocRewriter::getDenseResourceElementsAttrValue(mlir::PatternRewriter &rewriter, mlir::DenseResourceElementsAttr denseResourceAttr, 
    mlir::Type elementType, size_t flatIndex, mlir::Value &constVal) const {
    
     auto handle = denseResourceAttr.getRawHandle();
     auto blob = handle.getBlob();

     if (!blob) {
        llvm::errs() << "Error: Blob of DenseResourceElementsAttr is null!\n";
        return mlir::failure();
     }

     llvm::ArrayRef<char> data = blob->getData();
     unsigned bitWidth = elementType.getIntOrFloatBitWidth();

     if (elementType.isInteger()) {
        unsigned byteWidth = (bitWidth + 7) / 8;

        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(data.data());
        ptr += flatIndex * byteWidth;

        llvm::APInt value;
        if (bitWidth <= 64) {
            uint64_t rawValue = 0;
            memcpy(&rawValue, ptr, byteWidth);
            value = llvm::APInt(bitWidth, rawValue, elementType.isSignedInteger());
            auto elementAttr = rewriter.getIntegerAttr(elementType, value);
            constVal = rewriter.create<mlir::arith::ConstantOp>(rewriter.getUnknownLoc(), elementType, elementAttr);
        } else {
            llvm::errs() << "Error: Integer element type Bit width of DenseResourceElementsAttr is greater than 64!\n";
            return mlir::failure();
        }
     } else if (elementType.isFloat()) {
        if (bitWidth == 32) { 
            auto arrayRef = llvm::ArrayRef<float>(reinterpret_cast<const float *>(data.data()), data.size()/sizeof(float));
            auto elementAttr = rewriter.getF32FloatAttr(arrayRef[flatIndex]);
            constVal = rewriter.create<mlir::arith::ConstantOp>(rewriter.getUnknownLoc(), elementType, elementAttr);
        } else if (bitWidth == 64) {
            auto arrayRef = llvm::ArrayRef<double>(reinterpret_cast<const double *>(data.data()), data.size()/sizeof(double));
            auto elementAttr = rewriter.getF64FloatAttr(arrayRef[flatIndex]);
            constVal = rewriter.create<mlir::arith::ConstantOp>(rewriter.getUnknownLoc(), elementType, elementAttr);
        } else {
            llvm::errs() << "Error: Float element type Bit width of DenseResourceElementsAttr is not 32 or 64!\n";
            return mlir::failure();
        }
     } else {
        llvm::errs() << "Error: Element type of DenseResourceElementsAttr is not integer or float!\n";
        return mlir::failure();
     }


    return mlir::success();
}


mlir::LogicalResult 
GetGlobalToAllocRewriter::getDenseElementsAttrValue(mlir::PatternRewriter &rewriter, mlir::DenseElementsAttr denseAttr, 
    mlir::Type elementType, size_t flatIndex, mlir::Value &constVal) const {
    
    if (elementType.isInteger()) {
        auto initVals = denseAttr.getValues<llvm::APInt>();
        auto it = initVals.begin() + flatIndex;
        auto intAttr = rewriter.getIntegerAttr(elementType, *it);
        auto initValConstOp = rewriter.create<mlir::arith::ConstantOp>(rewriter.getUnknownLoc(), elementType, intAttr);
        constVal = initValConstOp.getResult();
    } else if (elementType.isFloat()) {
        auto initVals = denseAttr.getValues<llvm::APFloat>();
        auto it = initVals.begin() + flatIndex;
        auto floatAttr = rewriter.getFloatAttr(elementType, *it);
        auto initValConstOp = rewriter.create<mlir::arith::ConstantOp>(rewriter.getUnknownLoc(), elementType, floatAttr);
        constVal = initValConstOp.getResult();
    } else {
        llvm::errs() << "Error: Element type of dense attribute is not integer or float!\n";
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult GetGlobalToAllocRewriter::assignValueToMemref(mlir::PatternRewriter &rewriter, mlir::Value memref, size_t currentDim,
    mlir::Attribute initialValueAttr, llvm::ArrayRef<int64_t> &shape, llvm::SmallVector<int64_t> &indices) const {
        if (currentDim == shape.size()) {
            size_t flatIndex;
            mlir::Value initValConst;
            mlir::Type elementType = llvm::dyn_cast<mlir::MemRefType>(memref.getType()).getElementType();

            if (shape.size() == 0) {
                flatIndex = 0;
            } else if(shape.size() == 1) {
                flatIndex = indices[0];
            } else {
                flatIndex = 0;
                for (size_t i = 0; i < shape.size() - 1; i++) { 
                    flatIndex += indices[i] * shape[i];
                }
                flatIndex += indices[shape.size() - 1];
            }

            if (auto denseAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(initialValueAttr)) {
                if (mlir::failed(getDenseElementsAttrValue(rewriter, denseAttr, elementType, flatIndex, initValConst))) {
                    return mlir::failure();
                }
            } else if (auto resourceAttr = llvm::dyn_cast<mlir::DenseResourceElementsAttr>(initialValueAttr)) {
                if (mlir::failed(getDenseResourceElementsAttrValue(rewriter, resourceAttr, elementType, flatIndex, initValConst))) {
                    return mlir::failure();
                }
            } else {
                llvm::errs() << "Error: Initial value is not a supported dense attribute type!\n";
                return mlir::failure();
            }

            llvm::SmallVector<mlir::Value> indexValues;
            for (size_t i : indices) {
                auto idxConstOp = rewriter.create<mlir::arith::ConstantOp>(rewriter.getUnknownLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(i));
                indexValues.push_back(idxConstOp.getResult());
            }

            rewriter.create<mlir::affine::AffineStoreOp>(rewriter.getUnknownLoc(), initValConst, memref, indexValues);

            return mlir::success();
        }

        for (size_t i = 0; i < shape[currentDim]; i++) {
            indices.push_back(i);
            if (mlir::failed(assignValueToMemref(rewriter, memref, currentDim + 1, initialValueAttr, shape, indices))) {
                return mlir::failure();
            }
            indices.pop_back();
        }

        return mlir::success();
    }

void EquivFusionGetGlobalToAllocPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GetGlobalToAllocRewriter>(patterns.getContext());
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(), frozen))) {
        signalPassFailure();
    }
}

