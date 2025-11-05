#include "infrastructure/utils/namespace_macro.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

XUANSONG_NAMESPACE_HEADER_START


class EquivFusionManager {
private:
    static EquivFusionManager *instance_;

    mlir::OwningOpRef<mlir::ModuleOp> specModuleOp_;
    mlir::OwningOpRef<mlir::ModuleOp> implModuleOp_;

    mlir::OwningOpRef<mlir::ModuleOp> mergedModuleOp_;
    std::unique_ptr<mlir::MLIRContext> globalContext_;
    
    EquivFusionManager() = default;
    ~EquivFusionManager() = default;
    
    EquivFusionManager(const EquivFusionManager &) = delete;
    EquivFusionManager &operator=(const EquivFusionManager &) = delete;
    EquivFusionManager(EquivFusionManager &&) = delete;
    EquivFusionManager &operator=(EquivFusionManager &&) = delete;
    
public:
    static EquivFusionManager *getInstance();

    void setSpecModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getSpecModuleOp();
    void setImplModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getImplModuleOp();

    void setMergedModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getMergedModuleOp();
    mlir::MLIRContext* getGlobalContext();

};

XUANSONG_NAMESPACE_HEADER_END


