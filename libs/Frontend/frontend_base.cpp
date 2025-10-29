#include "libs/Frontend/frontend_base.h"

XUANSONG_NAMESPACE_HEADER_START

void FrontendBase::help(const std::string& name, const std::string& description) {
    log("\n");
    log("   overview: %s - %s\n", name.c_str(), description.c_str());
    log("   usage:    %s [-o filename]\n", name.c_str());
    log("   options:\n");
    log("       -o <filename>           - output filename\n");
    log("   example:");
    log("       %s -o test.output", name.c_str());
    log("\n\n");
}

bool FrontendBase::initOptions(const std::vector<std::string>& args, FrontendImplOptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto argv = args[idx];
        if (argv == "-o" && idx + 1 < args.size()) {
            opts.outputFilename = args[++idx];
        } else {
            opts.inputFilenames.emplace_back(argv);
        }
    }
    return true;
}

void FrontendBase::mergeModules(mlir::ModuleOp dest, mlir::ModuleOp src) {
    dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                           src.getBody()->getOperations());
}

XUANSONG_NAMESPACE_HEADER_END
