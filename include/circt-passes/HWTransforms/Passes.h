//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#ifndef EQUIVFUSION_HW_TRANSFORMS_PASSES_H
#define EQUIVFUSION_HW_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt {
namespace equivfusion {
namespace hw {

#define GEN_PASS_DECL
#include "circt-passes/HWTransforms/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt-passes/HWTransforms/Passes.h.inc"

}
}
}

#endif //EQUIVFUSION_HW_TRANSFORMS_PASSES_H
