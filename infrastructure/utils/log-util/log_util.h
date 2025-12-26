//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#pragma once

#include <cstdarg>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START

void log(const char *format, ...);

void logImpl(const char *format, va_list args);

void logWarning(const char *format, ...);

void logWarningImpl(const char *format, va_list args);

[[noreturn]] void logError(const char *format, ...);

[[noreturn]] void logErrorImpl(const char *format, va_list args);

void logEquivFusionBanner();

XUANSONG_NAMESPACE_HEADER_END