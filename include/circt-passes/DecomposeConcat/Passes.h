#ifndef EQUIVFUSION_DECOMPOSE_CONCAT_PASSES_H
#define EQUIVFUSION_DECOMPOSE_CONCAT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {

/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONDECOMPOSECONCAT
#define GEN_PASS_REGISTRATION
#include "circt-passes/DecomposeConcat/Passes.h.inc"

}

#endif // EQUIVFUSION_DECOMPOSE_CONCAT_PASSES_H
