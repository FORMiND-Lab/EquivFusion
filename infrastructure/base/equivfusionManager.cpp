#include "infrastructure/base/equivfusionManager.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"

#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"

XUANSONG_NAMESPACE_HEADER_START

EquivFusionManager *EquivFusionManager::instance_ = nullptr;

EquivFusionManager *EquivFusionManager::getInstance() {
    if (instance_ == nullptr) {
        instance_ = new EquivFusionManager();
    }
    return instance_;
}

void EquivFusionManager::setModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module) {
    moduleOp_ = std::move(module);
}
mlir::ModuleOp EquivFusionManager::getModuleOp() {
    return moduleOp_ ? moduleOp_.get() : nullptr;
}

mlir::MLIRContext* EquivFusionManager::getGlobalContext() {
    if (!globalContext_) {
        mlir::DialectRegistry registry;
        registry.insert<circt::comb::CombDialect>();
        registry.insert<circt::emit::EmitDialect>();
        registry.insert<circt::hw::HWDialect>();
        registry.insert<circt::om::OMDialect>();
        registry.insert<mlir::smt::SMTDialect>();
        registry.insert<circt::verif::VerifDialect>();
        registry.insert<mlir::func::FuncDialect>();

        mlir::func::registerInlinerExtension(registry);
        globalContext_ = std::make_unique<mlir::MLIRContext>(registry);
    }
    return globalContext_.get();
}

XUANSONG_NAMESPACE_HEADER_END

