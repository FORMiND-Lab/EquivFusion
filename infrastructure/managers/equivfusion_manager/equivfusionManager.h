#ifndef EQUIVFUSION_MANAGER_H
#define EQUIVFUSION_MANAGER_H

#include "infrastructure/utils/namespace_macro.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

XUANSONG_NAMESPACE_HEADER_START

enum class ModuleTypeEnum {
    UNKNOWN,
    SPEC,
    IMPL,
    MITER
};

class EquivFusionManager {
private:
    static EquivFusionManager *instance_;

    mlir::OwningOpRef<mlir::ModuleOp> specModuleOp_;
    mlir::OwningOpRef<mlir::ModuleOp> implModuleOp_;

    mlir::OwningOpRef<mlir::ModuleOp> mergedModuleOp_;
    std::unique_ptr<mlir::MLIRContext> globalContext_;

    std::set<std::string> inputPorts_;
    std::set<std::string> outputPorts_;
    
    EquivFusionManager() = default;
    ~EquivFusionManager() = default;
    
    EquivFusionManager(const EquivFusionManager &) = delete;
    EquivFusionManager &operator=(const EquivFusionManager &) = delete;
    EquivFusionManager(EquivFusionManager &&) = delete;
    EquivFusionManager &operator=(EquivFusionManager &&) = delete;
    
public:
    static EquivFusionManager *getInstance();

    void setSpecModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getSpecModuleOp() const;
    void setImplModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getImplModuleOp() const;

    void setMergedModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getMergedModuleOp() const;
    mlir::MLIRContext* getGlobalContext();

    void setModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module, ModuleTypeEnum moduleType);
    mlir::ModuleOp getModuleOp(ModuleTypeEnum moduleType);

    void addInputPorts(const std::vector<std::string>& ports);
    void addOutputPorts(const std::vector<std::string>& ports);
    void clearPorts();
    std::set<std::string> getInputPorts() const;
    std::set<std::string> getOutputPorts() const;

    void configureIRPrinting(mlir::PassManager &pm, bool enable);
};

XUANSONG_NAMESPACE_HEADER_END

#endif // EQUIVFUSION_MANAGER_H
