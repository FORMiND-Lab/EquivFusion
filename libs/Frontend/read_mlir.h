#ifndef EQUIVFUSION_READ_MLIR_H
#define EQUIVFUSION_READ_MLIR_H

#include "libs/Frontend/frontend_base.h"

XUANSONG_NAMESPACE_HEADER_START

class ReadMLIRImpl : public FrontendBase {
public:
    ReadMLIRImpl() = default;

public:
    static bool run(const std::vector<std::string>& args, 
                    mlir::MLIRContext& context,
                    mlir::ModuleOp module,
                    mlir::OwningOpRef<mlir::ModuleOp>& outputModule);
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_READ_MLIR_H
