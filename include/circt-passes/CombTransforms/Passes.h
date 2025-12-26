//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

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
