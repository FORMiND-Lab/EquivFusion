#ifndef EQUIVFUSION_WRITE_BTOR2_H
#define EQUIVFUSION_WRITE_BTOR2_H

#include "backend/backend_base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteBTOR2Impl : public BackendBase {
public:
    WriteBTOR2Impl() = default;

public:
    bool run(mlir::MLIRContext& context, mlir::ModuleOp module) override;
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_WRITE_BTOR2_H
