#ifndef EQUIVFUSION_BACKEND_H
#define EQUIVFUSION_BACKEND_H

#include <vector>
#include <string>
#include "infrastructure/utils/namespace_macro.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

XUANSONG_NAMESPACE_HEADER_START

class BackendBase {
public:
    BackendBase() = default;

    virtual bool initOptions(const std::vector<std::string>& args);
    virtual bool run(mlir::MLIRContext& context, mlir::ModuleOp module) = 0;

protected:
    std::string outputFilename_ {"-"};
};


XUANSONG_NAMESPACE_HEADER_END

#endif // EQUIVFUSION_BACKEND_H
