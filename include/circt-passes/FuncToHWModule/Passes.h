#pragma once

#include "mlir/Pass/Pass.h"

namespace circt {

/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONFUNCTOHWMODULE
#define GEN_PASS_REGISTRATION
#include "circt-passes/FuncToHWModule/Passes.h.inc"

}
