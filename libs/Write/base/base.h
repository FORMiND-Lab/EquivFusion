#ifndef EQUIVFUSION_WRITE_BASE_H
#define EQUIVFUSION_WRITE_BASE_H

#include <vector>
#include <string>
#include "infrastructure/utils/namespace_macro.h"


XUANSONG_NAMESPACE_HEADER_START

struct WriteImplOptions {
    std::string outputFilename {"-"};
};

class WriteBase {
public:
    static void help(const std::string& name, const std::string& description);

protected:
    static bool parseOptions(const std::vector<std::string>& args, WriteImplOptions& opts);

private:
    WriteBase() = default;
};


XUANSONG_NAMESPACE_HEADER_END

#endif // EQUIVFUSION_WRITE_BASE_H
