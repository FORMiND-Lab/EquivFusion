#include "libs/Backend/backend_base.h"

XUANSONG_NAMESPACE_HEADER_START

void BackendBase::help(const std::string& name, const std::string& description) {
        log("\n");
        log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());
        log("   USAGE:    %s [-o filename]\n", name.c_str());
        log("   OPTIONS:\n");
        log("       -o <filename>           - Output filename\n");
        log("   Example:");
        log("       %s -o test.output", name.c_str());
        log("\n\n");
}

bool BackendBase::parserOptions(const std::vector<std::string>& args, BackendImplOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "-o") {
            opts.outputFilename = args[++idx];
        }
    }
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
