#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"

#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

XUANSONG_NAMESPACE_HEADER_START

EquivFusionManager *EquivFusionManager::instance_ = nullptr;

EquivFusionManager *EquivFusionManager::getInstance() {
    if (instance_ == nullptr) {
        instance_ = new EquivFusionManager();
    }
    return instance_;
}

void EquivFusionManager::setSpecModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module) {
    specModuleOp_ = std::move(module);
}
mlir::ModuleOp EquivFusionManager::getSpecModuleOp() const {
    return specModuleOp_ ? specModuleOp_.get() : nullptr;
}
void EquivFusionManager::setImplModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module) {
    implModuleOp_ = std::move(module);
}
mlir::ModuleOp EquivFusionManager::getImplModuleOp() const {
    return implModuleOp_ ? implModuleOp_.get() : nullptr;
}

void EquivFusionManager::setMergedModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module) {
    mergedModuleOp_ = std::move(module);
}
mlir::ModuleOp EquivFusionManager::getMergedModuleOp() const {
    return mergedModuleOp_ ? mergedModuleOp_.get() : nullptr;
}

void EquivFusionManager::setModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module, ModuleTypeEnum moduleType) {
    switch (moduleType) {
        case ModuleTypeEnum::SPEC:
            setSpecModuleOp(module);
            break;
        case ModuleTypeEnum::IMPL:
            setImplModuleOp(module);
            break;
        case ModuleTypeEnum::MITER:
            setMergedModuleOp(module);
            break;
        case ModuleTypeEnum::UNKNOWN:
            assert(0 && "Invalid module type: UNKNOWN");
            break;
    }
}

mlir::ModuleOp EquivFusionManager::getModuleOp(ModuleTypeEnum moduleType) {
    switch (moduleType) {
        case ModuleTypeEnum::SPEC:
            return getSpecModuleOp();
        case ModuleTypeEnum::IMPL:
            return getImplModuleOp();
        case ModuleTypeEnum::MITER:
            return getMergedModuleOp();
        default:
            assert(0 && "Invalid module type");
            return nullptr;
    }
}

mlir::MLIRContext* EquivFusionManager::getGlobalContext() {
    if (!globalContext_) {
        mlir::DialectRegistry registry;
        registry.insert<circt::comb::CombDialect>();
        registry.insert<circt::emit::EmitDialect>();
        registry.insert<circt::hw::HWDialect>();
        registry.insert<circt::om::OMDialect>();
        registry.insert<circt::seq::SeqDialect>();
        registry.insert<circt::verif::VerifDialect>();
        registry.insert<circt::llhd::LLHDDialect>();
        registry.insert<circt::moore::MooreDialect>();

        registry.insert<mlir::smt::SMTDialect>();
        registry.insert<mlir::func::FuncDialect>();
        registry.insert<mlir::affine::AffineDialect>();
        registry.insert<mlir::memref::MemRefDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::cf::ControlFlowDialect>();
        registry.insert<mlir::scf::SCFDialect>();
        

        mlir::func::registerInlinerExtension(registry);
        globalContext_ = std::make_unique<mlir::MLIRContext>(registry);
        globalContext_->allowUnregisteredDialects();
    }
    return globalContext_.get();
}

void EquivFusionManager::addInputPorts(const std::vector<std::string>& ports) {
    inputPorts_.insert(ports.begin(), ports.end());
}

void EquivFusionManager::addOutputPorts(const std::vector<std::string>& ports) {
    outputPorts_.insert(ports.begin(), ports.end());
}

void EquivFusionManager::clearPorts() {
    inputPorts_.clear();
    outputPorts_.clear();
}

std::set<std::string> EquivFusionManager::getInputPorts() const {
    return inputPorts_;
}

std::set<std::string> EquivFusionManager::getOutputPorts() const {
    return outputPorts_;
}

void EquivFusionManager::configureIRPrinting(mlir::PassManager &pm, bool enable) {
    if (enable) {
        pm.enableIRPrinting(
                /*shouldPrintBeforePass=*/[](...) { return false; },
                /*shouldPrintAfterPass=*/[](...) { return true; },
                /*printModuleScope=*/false,
                /*printAfterOnlyOnChange=*/false,
                /*printAfterOnlyOnFailure=*/false
        );
    }
}

XUANSONG_NAMESPACE_HEADER_END

