#include "backend/backend_base.h"

XUANSONG_NAMESPACE_HEADER_START

bool BackendBase::initOptions(const std::vector<std::string>& args) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "-o") {
            outputFilename_ = args[++idx];
        }
    }
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
