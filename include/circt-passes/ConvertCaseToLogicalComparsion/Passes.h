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

namespace comb {

#define GEN_PASS_DECL_EQUIVFUSIONCONVERTCASETOLOGICALCOMPARSIONPASS
#define GEN_PASS_REGISTRATION

#include "circt-passes/ConvertCaseToLogicalComparsion/Passes.h.inc"

}

}
