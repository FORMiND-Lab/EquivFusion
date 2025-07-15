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