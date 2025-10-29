#ifndef EQUIVFUSION_FRONTEND_BASE_H
#define EQUIVFUSION_FRONTEND_BASE_H

#include <vector>
#include <string>
#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/log/log.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

XUANSONG_NAMESPACE_HEADER_START

struct FrontendImplOptions {
    std::vector<std::string> inputFilenames;
    std::string outputFilename;
};

class FrontendBase {
public:
    FrontendBase() = default;
    static void help(const std::string& name, const std::string& description);

protected:
    static bool initOptions(const std::vector<std::string>& args, FrontendImplOptions &opts);
    static void mergeModules(mlir::ModuleOp dest, mlir::ModuleOp src);
};


XUANSONG_NAMESPACE_HEADER_END

#endif // EQUIVFUSION_FRONTEND_BASE_H

