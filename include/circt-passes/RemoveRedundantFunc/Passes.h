#pragma once

#include "mlir/Pass/Pass.h"

namespace circt {

/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONREMOVEREDUNDANTFUNCPASS
#define GEN_PASS_REGISTRATION
#include "circt-passes/RemoveRedundantFunc/Passes.h.inc"
 
} // namespace circt
