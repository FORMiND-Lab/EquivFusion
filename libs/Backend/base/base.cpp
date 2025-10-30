#include "libs/Backend/base/base.h"

XUANSONG_NAMESPACE_HEADER_START

void BackendBase::help(const std::string& name, const std::string& description) {
        log("\n");
        log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());
        log("   USAGE:    %s [filename]\n", name.c_str());
        log("   Example:  %s test.output", name.c_str());
        log("\n\n");
}

bool BackendBase::parseOptions(const std::vector<std::string>& args, BackendImplOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        opts.outputFilename = arg;
    }
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
