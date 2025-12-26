//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EQUIVFUSION_UTILS_PATH_H
#define EQUIVFUSION_UTILS_PATH_H

#include <string>
#include <stdlib.h>

namespace XuanSong {
namespace Utils {
namespace PathUtil {

void expandTilde(std::string& path);

} // namespace PathUtil
} // namespace Utils
} // namespace XuanSong


#endif // EQUIVFUSION_UTILS_PATH_H
