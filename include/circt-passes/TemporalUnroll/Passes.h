#ifndef EQUIVFUSION_TEMPORAL_UNROLL_PASSES_H
#define EQUIVFUSION_TEMPORAL_UNROLL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {

/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONTEMPORALUNROLL
#define GEN_PASS_REGISTRATION
#include "circt-passes/TemporalUnroll/Passes.h.inc"

} // namespace circt

#endif //EQUIVFUSION_TEMPORAL_UNROLL_PASSES_H