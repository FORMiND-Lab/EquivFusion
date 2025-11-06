#include <iostream>
#include <cstdlib>
#include "infrastructure/utils/log/log.h"

XUANSONG_NAMESPACE_HEADER_START

void log(const char *format, ...) {
    va_list args;
    va_start(args, format);
    logImpl(format, args);
    va_end(args);
}

void logImpl(const char *format, va_list args) {
    vprintf(format, args);
}

void logWarning(const char *format, ...) {
    va_list args;
    va_start(args, format);
    logWarningImpl(format, args);
    va_end(args);
}


void logWarningImpl(const char *format, va_list args) {
    log("Warning: ");
    vprintf(format, args);
}

[[noreturn]] void logError(const char *format, ...) {
    va_list args;
    va_start(args, format);
    logErrorImpl(format, args);
    va_end(args);
}

[[noreturn]] void logErrorImpl(const char *format, va_list args) {
    log("Error: ");
    vprintf(format, args);
    exit(1);
}


void logEquivFusionBanner() {
    log(" /-------------------------------------------------------------------------\\\n");
    log(" |                                                                         |\n");
    log(" |               EquivFusion -- Silent as Pine, Precise as Logic           |\n");
    log(" |                                                                         |\n");
    log(" |                   Copyright (c) 2025 EquivFusion Team                   |\n");
    log(" |                           All rights reserved                           |\n");
    log(" |                                                                         |\n");
    log(" |                                                                         |\n");
    log(" \\-------------------------------------------------------------------------/\n");
    log("\n");
}

XUANSONG_NAMESPACE_HEADER_END