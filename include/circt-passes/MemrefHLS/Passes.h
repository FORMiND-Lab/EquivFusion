#pragma once

#include "mlir/Pass/Pass.h"

namespace circt {

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt-passes/MemrefHLS/Passes.h.inc"

}
