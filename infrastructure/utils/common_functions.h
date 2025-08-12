#pragma once

#include <string>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START

std::string nextToken(std::string &text, std::string sep);

void runCommand(std::string commandStr);

void runShell();

XUANSONG_NAMESPACE_HEADER_END

