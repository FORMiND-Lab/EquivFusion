//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "infrastructure/utils/path-util/path_util.h"
#include "infrastructure/utils/log-util/log_util.h"

#include "libs/Write/base/base.h"

XUANSONG_NAMESPACE_HEADER_START

void WriteBase::help(const std::string& name, const std::string& description) {
        log("\n");
        log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());
        log("   USAGE:    %s [filename]\n", name.c_str());
        log("   Example:  %s test.output", name.c_str());
        log("\n\n");
}

bool WriteBase::parseOptions(const std::vector<std::string>& args, WriteImplOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        opts.outputFilename = arg;
        Utils::PathUtil::expandTilde(opts.outputFilename);
    }
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
