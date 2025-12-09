#include "circt-passes/TemporalUnroll/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ValueMapper.h"
#include "mlir/IR/IRMapping.h"

namespace circt {
#define GEN_PASS_DEF_EQUIVFUSIONTEMPORALUNROLL
#include "circt-passes/TemporalUnroll/Passes.h.inc"
} // namespace circt;

using namespace circt;
using namespace mlir;

namespace {
struct EquivFusionTemporalUnrollPass
    : public circt::impl::EquivFusionTemporalUnrollBase<EquivFusionTemporalUnrollPass> {
    using circt::impl::EquivFusionTemporalUnrollBase<EquivFusionTemporalUnrollPass>::EquivFusionTemporalUnrollBase;

    void runOnOperation() override;

private:
    LogicalResult unrollModule(OpBuilder &builder, hw::HWModuleOp module);

    hw::HWModuleOp createUnrollModule(OpBuilder &builder, hw::HWModuleOp module);

    LogicalResult creatUnrollModuleBody(OpBuilder &builder, hw::HWModuleOp module, hw::HWModuleOp newModule);

    void initialRegistersResult(OpBuilder &builder, ValueMapper &currMapper, hw::HWModuleOp module);

    void prepareStepMapper(OpBuilder &builder, ValueMapper &currMapper,
                           hw::HWModuleOp module, hw::HWModuleOp newModule, unsigned step);

    void unrollFirRegOp(OpBuilder &builder, seq::FirRegOp op,
                        ValueMapper &prevMapper, ValueMapper &currMapper, unsigned step);

private:
    bool isClockPort(const StringRef &portName);
};
} // namespace


bool EquivFusionTemporalUnrollPass::isClockPort(const StringRef &portName) {
    return portName == "clk" || portName == "clock" ||
           portName == "CLK" || portName == "CLOCK";
}

LogicalResult EquivFusionTemporalUnrollPass::unrollModule(OpBuilder &builder, hw::HWModuleOp module) {
    // Step 1: Create new module
    auto newModule = createUnrollModule(builder, module);

    // Step 2: Cteate new module body
    if (failed(creatUnrollModuleBody(builder, module, newModule))) {
        return failure();
    }

    // Step 3: Remove original module
    module.erase();
    return success();
}

hw::HWModuleOp EquivFusionTemporalUnrollPass::createUnrollModule(OpBuilder &builder, hw::HWModuleOp module) {
    // Copy timeSteps ports of original module
    auto moduleType = module.getModuleType();
    SmallVector<hw::PortInfo> newPorts;
    for (unsigned step = 0; step < timeSteps; ++step) {
        for (const auto &port: moduleType.getPorts()) {
            if (isClockPort(port.name)) continue;  // skip clock port

            std::string newPortName = port.name.str() + "_" + std::to_string(step);
            newPorts.push_back({builder.getStringAttr(newPortName), port.type, port.dir});
        }
    }

    // Create new module
    std::string newModuleName = module.getName().str();
    auto newModule = hw::HWModuleOp::create(builder, module.getLoc(), builder.getStringAttr(newModuleName), newPorts);

    // Clear the automatically created body
    newModule.getBodyBlock()->clear();

    return newModule;
}

LogicalResult EquivFusionTemporalUnrollPass::creatUnrollModuleBody(OpBuilder &builder, hw::HWModuleOp module,
                                                                   hw::HWModuleOp newModule) {
    Block * oldBlock = module.getBodyBlock();
    Block * newBlock = newModule.getBodyBlock();
    builder.setInsertionPointToEnd(newBlock);

    BackedgeBuilder bb(builder, module.getLoc());
    ValueMapper prevMapper(&bb);
    ValueMapper currMapper(&bb);

    initialRegistersResult(builder, currMapper, module);

    SmallVector<Value> newOutputValues;;
    for (unsigned step = 0; step < timeSteps; ++step) {
        prevMapper = currMapper;
        prepareStepMapper(builder, currMapper, module, newModule, step);

        for (auto &op: oldBlock->getOperations()) {
            if (auto outputOp = dyn_cast < hw::OutputOp > (op)) {
                // OutputOp: collecting output values
                for (auto output: outputOp.getOperands())
                    newOutputValues.push_back(currMapper.get(output));
            } else if (auto firregOp = dyn_cast < seq::FirRegOp > (op)) {
                unrollFirRegOp(builder, firregOp, prevMapper, currMapper, step);
            } else {
                // Clone operation with mapping
                IRMapping bvMapper;
                for (auto operand: op.getOperands())
                    bvMapper.map(operand, currMapper.get(operand));
                auto *newOp = builder.clone(op, bvMapper);
                for (auto &&[oldRes, newRes]: llvm::zip(op.getResults(), newOp->getResults()))
                    currMapper.set(oldRes, newRes);
            }
        }
    }

    // Create final output operation
    hw::OutputOp::create(builder,newModule.getLoc(), newOutputValues);
    return success();
}

void EquivFusionTemporalUnrollPass::initialRegistersResult(OpBuilder &builder, ValueMapper &currMapper,
                                                           hw::HWModuleOp module) {
    // TODO(taomengxia): initial register result with zero
    Block * oldBlock = module.getBodyBlock();
    for (auto &op: oldBlock->getOperations()) {
        if (auto firregOp = dyn_cast < seq::FirRegOp > (op)) {
            auto initValue = hw::ConstantOp::create(builder, firregOp.getLoc(), firregOp.getType(), 0);
            currMapper.set(firregOp.getResult(), initValue);
        }
    }
}

void EquivFusionTemporalUnrollPass::prepareStepMapper(OpBuilder &builder, ValueMapper &currMapper,
                                                      hw::HWModuleOp module, hw::HWModuleOp newModule,
                                                      unsigned step) {
    Block * body = module.getBodyBlock();
    Block * newBody = newModule.getBodyBlock();
    auto moduleType = module.getModuleType();

    auto numArguments = body->getNumArguments();

    auto newNumArguments = newBody->getNumArguments();
    unsigned newArgIdx = newNumArguments / timeSteps * step;
    for (unsigned argIdx = 0; argIdx < numArguments; ++argIdx) {
        Value oldArg = body->getArgument(argIdx);

        auto pId = moduleType.getPortIdForInputId(argIdx);
        auto portName = moduleType.getPortName(pId);
        Value newArg;
        if (isClockPort(portName)) {
            newArg = hw::ConstantOp::create(builder, oldArg.getLoc(), oldArg.getType(), step % 2);
        } else {
            newArg = newBody->getArgument(newArgIdx++);
        }
        currMapper.set(oldArg, newArg);
    }
}

void EquivFusionTemporalUnrollPass::unrollFirRegOp(OpBuilder &builder, seq::FirRegOp regOp,
                                                   ValueMapper &prevMapper, ValueMapper &currMapper, unsigned step) {
    auto result = regOp.getResult();
    if (step == 0) {
        // currResult = hasReset : resetValue : initialValue
        auto currResult = regOp.hasReset() ? currMapper.get(regOp.getResetValue()) : prevMapper.get(result);
        currMapper.set(result, currResult);
        return;
    }

    auto loc = regOp.getLoc();

    auto clk = regOp.getClk();
    auto constOne = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    auto prevClk = seq::FromClockOp::create(builder, loc, prevMapper.get(clk));
    auto currClk = seq::FromClockOp::create(builder, loc, currMapper.get(clk));
    auto notPrevClk = comb::XorOp::create(builder, loc, prevClk, constOne);
    auto clockEdge = comb::AndOp::create(builder, loc, notPrevClk, currClk);

    auto prevResult = prevMapper.get(regOp.getResult());
    auto currNext = currMapper.get(regOp.getNext());

    Value currResult;
    if (regOp.hasReset()) {
        auto currReset = currMapper.get(regOp.getReset());
        auto currResetValue = currMapper.get(regOp.getResetValue());
        if (regOp.getIsAsync()) {
            //  currResult = (posedge reset) ? currResetValue :
            //                                 ((posedge clk) ? currNext : prevResult)
            auto prevReset = prevMapper.get(regOp.getReset());
            auto notPrevReset = comb::XorOp::create(builder, loc, prevReset, constOne);
            auto resetEdge = comb::AndOp::create(builder, loc, notPrevReset, currReset);

            currResult = comb::MuxOp::create(builder, loc, resetEdge,
                                             currResetValue,
                                             comb::MuxOp::create(builder, loc, clockEdge, currNext, prevResult));
        } else {
            // currResult = (posedge clk) ? ((currReset) ? currResetValue : currNext)) :
            //                                             prevResult
            currResult = comb::MuxOp::create(builder, loc, clockEdge,
                                             comb::MuxOp::create(builder, loc, currReset, currResetValue, currNext),
                                             prevResult);
        }
    } else {
        // currResult = (posedge clk) ? currNext : prevResult
        currResult = comb::MuxOp::create(builder, regOp.getLoc(), clockEdge, currNext, prevResult);
    }
    currMapper.set(result, currResult);
}

void EquivFusionTemporalUnrollPass::runOnOperation() {
    ModuleOp module = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());

    SmallVector<hw::HWModuleOp> modules;
    module.walk([&](hw::HWModuleOp module) {
        modules.push_back(module);
    });

    assert(modules.size() == 1);
    hw::HWModuleOp hwModule = modules[0];
    if (failed(unrollModule(builder, hwModule))) {
        return signalPassFailure();
    }
}