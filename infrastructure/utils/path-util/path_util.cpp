//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "infrastructure/utils/path-util/path_util.h"

namespace XuanSong {
namespace Utils {
namespace PathUtil {

void expandTilde(std::string& path) {
    if (path.compare(0, 2, "~/") == 0)
        path = path.replace(0, 1, getenv("HOME"));
}

} // namespace PathUtil
} // namespace Utils
} // namespace XuanSong
