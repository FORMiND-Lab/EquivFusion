//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#ifndef EQUIVFUSION_INITALLPASSES_H
#define EQUIVFUSION_INITALLPASSES_H

#include "circt-passes/CombTransforms/Passes.h"
#include "circt-passes/ConvertCaseToLogicalComparsion/Passes.h"
#include "circt-passes/FuncToHWModule/Passes.h"
#include "circt-passes/HWTransforms/Passes.h"
#include "circt-passes/MemrefHLS/Passes.h"
#include "circt-passes/Miter/Passes.h"
#include "circt-passes/RemoveRedundantFunc/Passes.h"
#include "circt-passes/TemporalUnroll/Passes.h"

namespace circt {
inline void registerAllEquivFusionPasses() {
    equivfusion::comb::registerPasses();
    comb::registerEquivFusionConvertCaseToLogicalComparisonPasses();
    registerEquivFusionFuncToHWModulePasses();
    equivfusion::hw::registerPasses();
    registerEquivFusionMemrefHLSPasses();
    equivfusion::registerEquivFusionMiterPasses();
    registerEquivFusionRemoveRedundantFuncPasses();
    equivfusion::registerEquivFusionTemporalUnrollPasses();
}

} // namespace circt

#endif //EQUIVFUSION_INITALLPASSES_H
