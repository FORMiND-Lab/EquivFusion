#ifndef EQUIVFUSION_WRITE_SMT_H
#define EQUIVFUSION_WRITE_SMT_H

#include "libs/Write/base/base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteSMTImpl final : public WriteBase {
public:
    static bool run(const std::vector<std::string>& args, mlir::MLIRContext& context,
                    mlir::ModuleOp inputModule, mlir::OwningOpRef<mlir::ModuleOp>& outputModule);

private:
    WriteSMTImpl() = default;
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_WRITE_SMT_H

