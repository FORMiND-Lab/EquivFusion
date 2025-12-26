//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/PrettyStackTrace.h"

#include "circt-passes/InitAllPasses.h"

int main(int argc, char **argv) {
    // Set the bug report message to indicate users should file issues on
    // llvm/circt and not llvm/llvm-project.
    llvm::setBugReportMsg(circt::circtBugReportMsg);

    mlir::DialectRegistry registry;

    // Register MLIR stuff
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<mlir::index::IndexDialect>();

    circt::registerAllDialects(registry);

    circt::registerAllEquivFusionPasses();

    return mlir::failed(mlir::MlirOptMain(
            argc, argv, "EquivFusion opt", registry));
}