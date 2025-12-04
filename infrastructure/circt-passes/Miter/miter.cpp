#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "circt-passes/Miter/Passes.h"


using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
namespace equivfusion {
#define GEN_PASS_DEF_EQUIVFUSIONMITER
#include "circt-passes/Miter/Passes.h.inc"
}
} // namespace circt


//===----------------------------------------------------------------------===//
// EquivFusionMiter pass
//===----------------------------------------------------------------------===//

namespace {
struct EquivFusionMiterPass
        : public circt::equivfusion::impl::EquivFusionMiterBase<EquivFusionMiterPass> {
    using circt::equivfusion::impl::EquivFusionMiterBase<EquivFusionMiterPass>::EquivFusionMiterBase;

    void runOnOperation() override;

private:
    hw::HWModuleOp lookupModule(StringRef name);

    llvm::LogicalResult
    constructMiter(OpBuilder &builder, Location loc, hw::HWModuleOp moduleA, hw::HWModuleOp moduleB);

    llvm::LogicalResult
    constructMiterForSMTLIB(OpBuilder &builder, Location loc, hw::HWModuleOp moduleA, hw::HWModuleOp moduleB);

    llvm::LogicalResult
    constructMiterForAIGER(OpBuilder &builder, Location loc, hw::HWModuleOp moduleA, hw::HWModuleOp moduleB);

    llvm::LogicalResult
    constructMiterForBTOR2(OpBuilder &builder, Location loc, hw::HWModuleOp moduleA, hw::HWModuleOp moduleB);

    std::pair<hw::HWModuleOp, Value> createTopModule(OpBuilder& builder, Location loc,
                                                     hw::HWModuleOp moduleA, hw::HWModuleOp moduleB);
};
} // namespace

hw::HWModuleOp EquivFusionMiterPass::lookupModule(StringRef name) {
    Operation *expectedModule = SymbolTable::lookupNearestSymbolFrom(
            getOperation(), StringAttr::get(&getContext(), name));
    if (!expectedModule || !isa<hw::HWModuleOp>(expectedModule)) {
        getOperation().emitError("module named '") << name << "' not found";
        return {};
    }
    return cast<hw::HWModuleOp>(expectedModule);
}

llvm::LogicalResult EquivFusionMiterPass::constructMiter(OpBuilder &builder, Location loc,
                                                         hw::HWModuleOp moduleA,
                                                         hw::HWModuleOp moduleB) {
    switch (miterMode) {
        case circt::equivfusion::MiterModeEnum::SMTLIB:
            return constructMiterForSMTLIB(builder, loc, moduleA, moduleB);
        case circt::equivfusion::MiterModeEnum::AIGER:
            return constructMiterForAIGER(builder, loc, moduleA, moduleB);
        case circt::equivfusion::MiterModeEnum::BTOR2:
            return constructMiterForBTOR2(builder, loc, moduleA, moduleB);
        default:
            return llvm::failure();
    }
}

llvm::LogicalResult EquivFusionMiterPass::constructMiterForSMTLIB(OpBuilder &builder, Location loc,
                                                                  hw::HWModuleOp moduleA,
                                                                  hw::HWModuleOp moduleB) {
    auto lecOp = builder.create<verif::LogicEquivalenceCheckingOp>(loc, false);
    builder.cloneRegionBefore(moduleA.getBody(), lecOp.getFirstCircuit(),
                              lecOp.getFirstCircuit().end());
    builder.cloneRegionBefore(moduleB.getBody(), lecOp.getSecondCircuit(),
                              lecOp.getSecondCircuit().end());

    moduleA->erase();
    if (moduleA != moduleB)
        moduleB->erase();

    {
        auto *term = lecOp.getFirstCircuit().front().getTerminator();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(term);
        builder.create<verif::YieldOp>(loc, term->getOperands());
        term->erase();
        term = lecOp.getSecondCircuit().front().getTerminator();
        builder.setInsertionPoint(term);
        builder.create<verif::YieldOp>(loc, term->getOperands());
        term->erase();
    }

    sortTopologically(&lecOp.getFirstCircuit().front());
    sortTopologically(&lecOp.getSecondCircuit().front());

    return llvm::success();
}

llvm::LogicalResult EquivFusionMiterPass::constructMiterForAIGER(OpBuilder &builder, Location loc,
                                                                 hw::HWModuleOp moduleA,
                                                                 hw::HWModuleOp moduleB) {
    /// Create topModule.
    auto [topModule, outputsEq] = createTopModule(builder, loc, moduleA, moduleB);
    if (!topModule) {
        return llvm::failure();
    }

    /// Construct neq
    Value zero = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 0);
    Value outputsNeq = builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, outputsEq, zero);

    auto *term = topModule.getBodyBlock()->getTerminator();
    builder.create<hw::OutputOp>(loc, outputsNeq);
    term->erase();

    return llvm::success();
}

llvm::LogicalResult EquivFusionMiterPass::constructMiterForBTOR2(OpBuilder &builder, Location loc,
                                                                 hw::HWModuleOp moduleA,
                                                                 hw::HWModuleOp moduleB) {
    /// Create topModule.
    auto [topModule, outputsEq] = createTopModule(builder, loc, moduleA, moduleB);
    if (!topModule) {
        return llvm::failure();
    }

    /// Construct verif assert.
    builder.create<verif::AssertOp>(loc, outputsEq, Value{}, StringAttr{});

    auto *term = topModule.getBodyBlock()->getTerminator();
    builder.create<hw::OutputOp>(loc, outputsEq);
    term->erase();

    return llvm::success();
}

std::pair<hw::HWModuleOp, Value> EquivFusionMiterPass::createTopModule(OpBuilder &builder, Location loc,
                                                                       hw::HWModuleOp moduleA,
                                                                       hw::HWModuleOp moduleB) {
    auto moduleAType = moduleA.getModuleType();
    SmallVector<hw::PortInfo> ports;
    for (auto port: moduleAType.getPorts()) {
        if (port.dir == hw::ModulePort::Direction::Input) {
            ports.push_back({port.name, port.type, port.dir});
        }
    }

    hw::PortInfo miterPort = {builder.getStringAttr("__output"),
                              builder.getI1Type(),
                              hw::ModulePort::Direction::Output};
    ports.push_back(miterPort);

    /// Create topModule 
    auto topModule = builder.create<hw::HWModuleOp>(loc, builder.getStringAttr("__Top"), ports);
    builder.setInsertionPointToStart(topModule.getBodyBlock());

    Block * bodyBlock = topModule.getBodyBlock();
    SmallVector<Value> instanceInputs(bodyBlock->args_begin(), bodyBlock->args_end());

    /// Instantiate moduleA/moduleB
    auto instanceA = builder.create<hw::InstanceOp>(loc, moduleA, builder.getStringAttr("instanceA"), instanceInputs);
    auto instanceB = builder.create<hw::InstanceOp>(loc, moduleB, builder.getStringAttr("instanceB"), instanceInputs);
    assert(instanceA.getNumResults() == instanceB.getNumResults() &&
           "Modules must have the same number of outputs.");

    /// Set private for flatten
    moduleA.setPrivate();
    moduleB.setPrivate();

    /// Construct Equal
    unsigned outputNum = instanceA.getNumResults();
    SmallVector<Value> outputsEqual;
    for (unsigned i = 0; i < outputNum; ++i) {
        Value o1 = instanceA.getResult(i);
        Value o2 = instanceB.getResult(i);

        if (isHWIntegerType(o1.getType())) {
            outputsEqual.emplace_back(builder.create<comb::ICmpOp>(loc, comb::ICmpPredicate::eq, o1, o2));
        } else {
            moduleA.emitError("[EquivFusionMiterPass] : unsupported output type ") << o1.getType();
            return {nullptr, nullptr};
        }
    }

    Value equal;
    switch (outputNum) {
        case 0:
            equal = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 1);
            break;
        case 1:
            equal = outputsEqual[0];
            break;
        default:
            equal = builder.create<comb::AndOp>(loc, outputsEqual, false);
            break;
    }

    return {topModule, equal};
}

void EquivFusionMiterPass::runOnOperation() {
    OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
    Location loc = getOperation()->getLoc();

    // Lookup the modules
    auto moduleA = lookupModule(firstModule);
    if (!moduleA)
        return signalPassFailure();
    auto moduleB = lookupModule(secondModule);
    if (!moduleB)
        return signalPassFailure();

    if (moduleA.getModuleType() != moduleB.getModuleType()) {
        moduleA.emitError("module's IO types don't match second modules: ")
                << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
        return signalPassFailure();
    }

    if (failed(constructMiter(builder, loc, moduleA, moduleB))) {
        return signalPassFailure();
    }
}
