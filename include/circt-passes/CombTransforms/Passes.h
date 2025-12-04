#ifndef EQUIVFUSION_COMB_TRANSFORMS_PASSES_H
#define EQUIVFUSION_COMB_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt {
namespace equivfusion {
namespace comb {

#define GEN_PASS_DECL
#include "circt-passes/CombTransforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt-passes/CombTransforms/Passes.h.inc"

}
}
}

#endif // EQUIVFUSION_COMB_TRANSFORMS_PASSES_H
