//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#ifndef EQUIVFUSION_TEMPORAL_UNROLL_PASSES_H
#define EQUIVFUSION_TEMPORAL_UNROLL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt {
namespace equivfusion {
/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONTEMPORALUNROLL
#define GEN_PASS_REGISTRATION

#include "circt-passes/TemporalUnroll/Passes.h.inc"

} // namespace equivfusion
} // namespace circt

#endif //EQUIVFUSION_TEMPORAL_UNROLL_PASSES_H