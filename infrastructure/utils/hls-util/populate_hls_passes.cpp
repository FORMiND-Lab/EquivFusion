#include "infrastructure/utils/hls-util/populate_hls_passes.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Transforms/Passes.h"


#include "circt-passes/FuncToHWModule/Passes.h"
#include "circt-passes/RemoveRedundantFunc/Passes.h"
#include "circt-passes/MemrefHLS/Passes.h"

XUANSONG_NAMESPACE_HEADER_START

void populateHLSPasses(mlir::PassManager &pm) {
    // Unroll affine loops.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopUnrollPass(-1, false, [](mlir::affine::AffineForOp forOp) -> unsigned int {
        std::optional<uint64_t> mayBeConstantTripCount = mlir::affine::getConstantTripCount(forOp);
        if (mayBeConstantTripCount) {
            return *mayBeConstantTripCount;
        }
        return 0;
    }));
    
    // Convert 'memref.get_global' to 'memref.alloc'
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(circt::createEquivFusionGetGlobalToAllocPass());

    // Replace affine memref accesses by scalars by forwarding stores to loads and eliminating redundant loads.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineScalarReplacementPass());

    // Lower affine to a conbination of Arith and
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLowerAffinePass());

    // Convert multi-dimensional memref to one-dimensional memref.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(circt::createFlattenMemRefPass());
    pm.addPass(circt::createFlattenMemRefCallsPass());

    pm.enableVerifier(false);
    // Convert all index typed values to an i32 integer.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::func::FuncOp>(circt::createEquivFusionConvertIndexToI32Pass());

    // Lower Scf to ControlFlow dialect.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCFToControlFlowPass());

    // Convert Arith to Comb
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(circt::createMapArithToCombPass());

    //Convert Func to Module
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(circt::createEquivFusionFuncToHWModule());
    
    pm.addPass(mlir::createCanonicalizerPass());
}

XUANSONG_NAMESPACE_HEADER_END