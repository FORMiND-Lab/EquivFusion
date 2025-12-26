//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "infrastructure/utils/string-util/string_util.h"

XUANSONG_NAMESPACE_HEADER_START
namespace Utils {
namespace StringUtil {

std::string nextToken(std::string &text, std::string sep) {
    size_t beginPos = text.find_first_not_of(sep);
    if (beginPos == std::string::npos) {
        return "";
    }

    size_t endPos = text.find_first_of(sep, beginPos);
    if (endPos == std::string::npos) {
        endPos = text.size();
    }

    std::string token = text.substr(beginPos, endPos - beginPos);
    text = text.substr(endPos);
    return token;
}

} // namespace StringUtil
} // namespace Utils
XUANSONG_NAMESPACE_HEADER_END

