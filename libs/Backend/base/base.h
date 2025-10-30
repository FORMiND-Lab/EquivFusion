#ifndef EQUIVFUSION_BACKEND_H
#define EQUIVFUSION_BACKEND_H

#include <vector>
#include <string>
#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/log/log.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

XUANSONG_NAMESPACE_HEADER_START

struct BackendImplOptions {
    std::string outputFilename {"-"};
};

class BackendBase {
public:
    static void help(const std::string& name, const std::string& description);

protected:
    static bool parseOptions(const std::vector<std::string>& args, BackendImplOptions& opts);

private:
    BackendBase() = default;
};


XUANSONG_NAMESPACE_HEADER_END

#endif // EQUIVFUSION_BACKEND_H
