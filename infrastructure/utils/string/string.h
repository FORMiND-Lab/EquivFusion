#pragma once

#include <string>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START
namespace Utils {
namespace StringUtil {

std::string nextToken(std::string &text, std::string sep);

} // namespace StringUtil
} // namespace Utils
XUANSONG_NAMESPACE_HEADER_END

