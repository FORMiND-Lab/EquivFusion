#ifndef EQUIVFUSION_WRITE_BTOR2_H
#define EQUIVFUSION_WRITE_BTOR2_H

#include "libs/Backend/backend_base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteBTOR2Impl : public BackendBase {
public:
    WriteBTOR2Impl() = default;

public:
    static bool run(const std::vector<std::string>&args,
                    mlir::MLIRContext& context, mlir::ModuleOp inputModule,
                    mlir::OwningOpRef<mlir::ModuleOp>& outputModule);
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_WRITE_BTOR2_H
