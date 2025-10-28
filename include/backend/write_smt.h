#ifndef EQUIVFUSION_WRITE_SMT_H
#define EQUIVFUSION_WRITE_SMT_H

#include "backend/backend_base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteSMTImpl : public BackendBase {
public:
    WriteSMTImpl() = default;

public:
    bool run(mlir::MLIRContext& context, mlir::ModuleOp module) override;
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_WRITE_SMT_H

