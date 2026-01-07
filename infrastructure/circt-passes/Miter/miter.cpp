//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "circt-passes/Miter/Passes.h"
#include "llvm/ADT/StringMap.h"


using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
namespace equivfusion {
#define GEN_PASS_DEF_EQUIVFUSIONMITER
#include "circt-passes/Miter/Passes.h.inc"
}
} // namespace circt


namespace {
struct PortPerimutation {
    SmallVector<unsigned> inIndexMap;
    SmallVector<unsigned> outIndexMap;
};
} // namespace


static FailureOr<PortPerimutation> computePortPermutationByName(hw::HWModuleOp moduleA, hw::HWModuleOp moduleB) {
    auto moduleAType = moduleA.getModuleType();
    auto moduleBType = moduleB.getModuleType();

    if (moduleAType.getNumInputs() != moduleBType.getNumInputs() || moduleAType.getNumOutputs() != moduleBType.getNumOutputs()) {
        moduleA.emitError("module's IO types don't match second modules: ")
                << moduleA.getModuleType() << " vs " << moduleB.getModuleType();

        return llvm::failure();
    }

    std::map<std::string, int> moduleAInputNameToIndexMap;
    std::map<std::string, int> moduleAOutputNameToIndexMap;

    unsigned inIndex = 0;
    unsigned outIndex = 0;

    for (auto port: moduleAType.getPorts()) {
        if (port.dir == hw::ModulePort::Direction::Input) {
            moduleAInputNameToIndexMap[port.name.str()] = inIndex++;
        } else if (port.dir == hw::ModulePort::Direction::Output) {
            moduleAOutputNameToIndexMap[port.name.str()] = outIndex++;
        }
    }

    PortPerimutation perm;
    for (auto port: moduleBType.getPorts()) {
        std::string portName = port.name.str();

        if (port.dir == hw::ModulePort::Direction::Input) {
            if ((moduleAInputNameToIndexMap.find(portName) == moduleAInputNameToIndexMap.end()) || 
                (moduleAType.getInputType(moduleAInputNameToIndexMap[portName]) != port.type)) {
                moduleA.emitError("module's IO types don't match second modules: ")
                        << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
                return llvm::failure();
            }

            perm.inIndexMap.push_back(moduleAInputNameToIndexMap[portName]);
        } else if (port.dir == hw::ModulePort::Direction::Output) {
            if ((moduleAOutputNameToIndexMap.find(portName) == moduleAOutputNameToIndexMap.end()) || 
                (moduleAType.getOutputType(moduleAOutputNameToIndexMap[portName]) != port.type)) {
                moduleA.emitError("module's IO types don't match second modules: ")
                        << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
                return llvm::failure();
            }

            perm.outIndexMap.push_back(moduleAOutputNameToIndexMap[portName]);
        }
    }

    return perm;
}


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
    auto permOrFail = computePortPermutationByName(moduleA, moduleB);
    if (failed(permOrFail)) {
        return llvm::failure();
    }
    PortPerimutation perm = *permOrFail;

    auto lecOp = verif::LogicEquivalenceCheckingOp::create(builder, loc, false);
    builder.cloneRegionBefore(moduleA.getBody(), lecOp.getFirstCircuit(),
                              lecOp.getFirstCircuit().end());
                              
    Region &srcRegion = moduleB.getBody();
    Block &srcBlock = srcRegion.front();
    auto aType = moduleA.getModuleType();

    Region &dstRegion = lecOp.getSecondCircuit();
    Block *dstBlock = new Block();
    dstRegion.push_back(dstBlock);

    sortTopologically(&srcBlock);

    for (unsigned i = 0, e = aType.getNumInputs(); i < e; ++i)
        dstBlock->addArgument(aType.getInputType(i), loc);

    IRMapping mapper;
    for (unsigned bArg = 0, e = srcBlock.getNumArguments(); bArg < e; ++bArg) {
        unsigned aArg = perm.inIndexMap[bArg];
        mapper.map(srcBlock.getArgument(bArg), dstBlock->getArgument(aArg));
    }

    builder.setInsertionPointToEnd(dstBlock);
    for (auto& op : llvm::make_early_inc_range(srcBlock.without_terminator())) {
        mlir::Operation* clonedOp = builder.clone(op, mapper);

        if (!clonedOp) {
            llvm::errs() << "Failed to clone operation " << op << "\n";
            return mlir::failure();
          }
    }
        
    SmallVector<Value> outReordered;
    outReordered.resize(aType.getNumOutputs());
    auto *termOp = srcBlock.getTerminator();


    for (unsigned i = 0; i < aType.getNumOutputs(); ++i) {
        outReordered[perm.outIndexMap[i]] = mapper.lookupOrDefault(termOp->getOperand(i));
    }

    builder.setInsertionPoint(dstBlock, dstBlock->end());
    hw::OutputOp::create(builder, loc, outReordered);

    moduleA->erase();
    if (moduleA != moduleB)
        moduleB->erase();

    {
        auto *term = lecOp.getFirstCircuit().front().getTerminator();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(term);
        verif::YieldOp::create(builder, loc, term->getOperands());
        term->erase();
        term = lecOp.getSecondCircuit().front().getTerminator();
        builder.setInsertionPoint(term);
        verif::YieldOp::create(builder, loc, term->getOperands());
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
    Value zero = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
    Value outputsNeq = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq, outputsEq, zero);

    auto *term = topModule.getBodyBlock()->getTerminator();
    hw::OutputOp::create(builder, loc, outputsNeq);
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
    verif::AssertOp::create(builder, loc, outputsEq, Value{}, StringAttr{});

    auto *term = topModule.getBodyBlock()->getTerminator();
    hw::OutputOp::create(builder, loc, outputsEq);
    term->erase();

    return llvm::success();
}

std::pair<hw::HWModuleOp, Value> EquivFusionMiterPass::createTopModule(OpBuilder &builder, Location loc,
                                                                       hw::HWModuleOp moduleA,
                                                                       hw::HWModuleOp moduleB) {
    FailureOr<PortPerimutation> permOrFail = computePortPermutationByName(moduleA, moduleB);
    if (failed(permOrFail)) {
        return {nullptr, nullptr};
    }
    PortPerimutation perm = *permOrFail;

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
    auto topModule = hw::HWModuleOp::create(builder, loc, builder.getStringAttr("__Top"), ports);
    builder.setInsertionPointToStart(topModule.getBodyBlock());

    Block * bodyBlock = topModule.getBodyBlock();
    SmallVector<Value> instanceAInputs(bodyBlock->args_begin(), bodyBlock->args_end());
    SmallVector<Value> instanceBInputs;

    for (auto i = 0; i < perm.inIndexMap.size(); ++i) {
        instanceBInputs.push_back(instanceAInputs[perm.inIndexMap[i]]);
    }

    /// Instantiate moduleA/moduleB
    auto instanceA = hw::InstanceOp::create(builder, loc, moduleA, builder.getStringAttr("instanceA"), instanceAInputs);
    auto instanceB = hw::InstanceOp::create(builder, loc, moduleB, builder.getStringAttr("instanceB"), instanceBInputs);
    assert(instanceA.getNumResults() == instanceB.getNumResults() &&
           "Modules must have the same number of outputs.");

    /// Set private for flatten
    moduleA.setPrivate();
    moduleB.setPrivate();

    /// Construct Equal
    unsigned outputNum = instanceA.getNumResults();
    SmallVector<Value> outputsEqual;
    for (unsigned i = 0; i < outputNum; ++i) {
        Value o1 = instanceA.getResult(perm.outIndexMap[i]);
        Value o2 = instanceB.getResult(i);

        if (isHWIntegerType(o1.getType())) {
            outputsEqual.emplace_back(comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq, o1, o2));
        } else {
            moduleA.emitError("[EquivFusionMiterPass] : unsupported output type ") << o1.getType();
            return {nullptr, nullptr};
        }
    }

    Value equal;
    switch (outputNum) {
        case 0:
            equal = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
            break;
        case 1:
            equal = outputsEqual[0];
            break;
        default:
            equal = comb::AndOp::create(builder, loc, outputsEqual, false);
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

/*
    if (moduleA.getModuleType() != moduleB.getModuleType()) {
        moduleA.emitError("module's IO types don't match second modules: ")
                << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
        return signalPassFailure();
    }
*/
    if (failed(constructMiter(builder, loc, moduleA, moduleB))) {
        return signalPassFailure();
    }
}
