#ifndef EQUIVFUSION_WRITE_AIGER_H
#define EQUIVFUSION_WRITE_AIGER_H

#include "libs/Write/base/base.h"

XUANSONG_NAMESPACE_HEADER_START

class WriteAIGERImpl final : public WriteBase {
public:
    static bool run(const std::vector<std::string>& args, mlir::MLIRContext& context, mlir::ModuleOp inputModule);

private:
    WriteAIGERImpl() = default;
};


XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong


#endif // EQUIVFUSION_WRITE_AIGER_H

