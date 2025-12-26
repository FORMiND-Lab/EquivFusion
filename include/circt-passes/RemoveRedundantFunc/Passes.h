//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"

namespace circt {

/// Generate the code for registering passes.
#define GEN_PASS_DECL_EQUIVFUSIONREMOVEREDUNDANTFUNCPASS
#define GEN_PASS_REGISTRATION
#include "circt-passes/RemoveRedundantFunc/Passes.h.inc"
 
} // namespace circt
