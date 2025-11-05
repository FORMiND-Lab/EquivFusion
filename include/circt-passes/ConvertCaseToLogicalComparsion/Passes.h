#pragma once

#include "mlir/Pass/Pass.h"

namespace circt {

namespace comb {

#define GEN_PASS_DECL_EQUIVFUSIONCONVERTCASETOLOGICALCOMPARSIONPASS
#define GEN_PASS_REGISTRATION

#include "circt-passes/ConvertCaseToLogicalComparsion/Passes.h.inc"

}

}
