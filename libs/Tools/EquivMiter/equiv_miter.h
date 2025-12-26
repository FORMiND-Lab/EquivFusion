//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EQUIVFUSION_EQUIV_MITER_H
#define EQUIVFUSION_EQUIV_MITER_H

#include <vector>
#include <string>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START
namespace EquivMiterTool {

void help(const std::string &name, const std::string &description);
bool execute(const std::vector<std::string> &args);

}  // namespace EquivMiterTool
XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong

#endif //EQUIVFUSION_EQUIV_MITER_H
