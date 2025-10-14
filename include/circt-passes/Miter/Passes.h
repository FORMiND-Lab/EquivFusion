//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for equivfusion miter passes.
//
//===----------------------------------------------------------------------===//


#ifndef EQUIVFUSION_MITER_PASSES_H
#define EQUIVFUSION_MITER_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {
namespace EquivFusionMiter {

enum class MiterModeEnum {
    /// Miter for SMTLIB output
    SMTLIB,

    /// Miter for AIGER output
    AIGER,

    /// Miter for BTOR2 output
    BTOR2,
};

}

/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONMITER
#define GEN_PASS_REGISTRATION
#include "circt-passes/Miter/Passes.h.inc"

}


#endif // EQUIVFUSION_MITER_PASSES_H
