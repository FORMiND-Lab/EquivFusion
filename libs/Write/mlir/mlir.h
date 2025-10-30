#ifndef EQUIVFUSION_WRITE_MLIR_IMPL_H
#define EQUIVFUSION_WRITE_MLIR_IMPL_H

#include "libs/Write/base/base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteMLIRImpl final : public WriteBase {
public:
    static bool run(const std::vector<std::string>& args, mlir::MLIRContext& context, mlir::ModuleOp inputModule);

private:
    WriteMLIRImpl() = default;
};

XUANSONG_NAMESPACE_HEADER_END

#endif // EQUIVFUSION_WRITE_MLIR_IMPL_H
