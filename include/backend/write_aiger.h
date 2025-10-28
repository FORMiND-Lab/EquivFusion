#ifndef EQUIVFUSION_WRITE_AIGER_H
#define EQUIVFUSION_WRITE_AIGER_H

#include "backend/backend_base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteAIGERImpl : public BackendBase {
public:
    WriteAIGERImpl() = default;

public:
    bool run(mlir::MLIRContext& context, mlir::ModuleOp module) override;
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_WRITE_AIGER_H

