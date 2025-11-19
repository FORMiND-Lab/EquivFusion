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

	void initialRegistersResult(OpBuilder &builder, hw::HWModuleOp module, ValueMapper &currMapper);
	void prepareStepMapper(hw::HWModuleOp module, hw::HWModuleOp newModule,
				           ValueMapper &prevMapper, ValueMapper &currMapper, unsigned step);
	void unrollFirRegOp(OpBuilder &builder, seq::FirRegOp op,
	                    ValueMapper &prevMapper, ValueMapper &currMapper, unsigned step);
};
} // namespace


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
		for (const auto &port : moduleType.getPorts()) {
			std::string newPortName = port.name.str() + "_" + std::to_string(step);
			newPorts.push_back({builder.getStringAttr(newPortName), port.type, port.dir});
		}
	}

	// Create new module
	std::string newModuleName = module.getName().str();
    auto newModule = builder.create<hw::HWModuleOp>(module.getLoc(), builder.getStringAttr(newModuleName), newPorts);

	// Clear the automatically created body
	newModule.getBodyBlock()->clear();

	return newModule;
}

LogicalResult EquivFusionTemporalUnrollPass::creatUnrollModuleBody(OpBuilder &builder, hw::HWModuleOp module,
																	hw::HWModuleOp newModule) {
	Block *newBlock = newModule.getBodyBlock();
	builder.setInsertionPointToEnd(newBlock);

	BackedgeBuilder bb(builder, module.getLoc());
	ValueMapper prevMapper(&bb);
	ValueMapper currMapper(&bb);

	initialRegistersResult(builder, module, currMapper);

	SmallVector<Value> newOutputValues;;
	Block *oldBlock = module.getBodyBlock();
	for (unsigned step = 0; step < timeSteps; ++step) {
		prepareStepMapper(module, newModule, prevMapper, currMapper, step);

		for (auto &op : oldBlock->getOperations()) {
			if (auto outputOp = dyn_cast<hw::OutputOp>(op)) {
				// OutputOp: collecting output values
				for (auto output : outputOp.getOperands())
					newOutputValues.push_back(currMapper.get(output));
			} else if (auto firregOp = dyn_cast<seq::FirRegOp>(op)) {
				unrollFirRegOp(builder, firregOp, prevMapper, currMapper, step);
			} else {
				// Clone operation with mapping
				IRMapping bvMapper;
				for (auto operand : op.getOperands())
					bvMapper.map(operand, currMapper.get(operand));
				auto *newOp = builder.clone(op, bvMapper);
				for (auto &&[oldRes, newRes] : llvm::zip(op.getResults(), newOp->getResults()))
					currMapper.set(oldRes, newRes);
			}
		}
	}

	// Create final output operation
	builder.create<hw::OutputOp>(newModule.getLoc(), newOutputValues);
	return success();
}

void EquivFusionTemporalUnrollPass::initialRegistersResult(OpBuilder &builder, hw::HWModuleOp module,
                                                             ValueMapper &currMapper) {
	// TODO(taomengxia): initial register result with zero
	Block *oldBlock = module.getBodyBlock();
	for (auto &op : oldBlock->getOperations()) {
		if (auto firregOp = dyn_cast<seq::FirRegOp>(op)) {
			auto initValue = hw::ConstantOp::create(builder, firregOp.getLoc(), firregOp.getType(), 0);
			currMapper.set(firregOp.getResult(), initValue);
		}
	}
}

void EquivFusionTemporalUnrollPass::prepareStepMapper(hw::HWModuleOp module, hw::HWModuleOp newModule,
                                                      ValueMapper &prevMapper, ValueMapper& currMapper, unsigned step) {
	prevMapper = currMapper;

	Block *oldBody = module.getBodyBlock();
	Block *newBody = newModule.getBodyBlock();

	auto oldArgumentsSize = oldBody->getArguments().size();
	for (unsigned argIdx = 0; argIdx < oldArgumentsSize; ++argIdx) {
		Value oldArgValue = oldBody->getArgument(argIdx);
		Value newArgValue = newBody->getArgument(argIdx + step * oldArgumentsSize);
		currMapper.set(oldArgValue, newArgValue);
 	}
}

void EquivFusionTemporalUnrollPass::unrollFirRegOp(OpBuilder &builder, seq::FirRegOp op,
                                                   ValueMapper &prevMapper, ValueMapper& currMapper, unsigned step) {
	// TODO (taomengxia): reset value
	auto next = op.getNext();
	auto clk = op.getClk();
	auto result = op.getResult();

	if (step == 0) {
		// TODO(taomengxia): initial value
		// currResult = currNext
		currMapper.set(result, currMapper.get(next));
	} else {
		// currResult = (!prevClk && currClk) ? currNext : prevResult
		Value constOne = hw::ConstantOp::create(builder, op.getLoc(), builder.getI1Type(), 1);
		Value prevClk = seq::FromClockOp::create(builder, op.getLoc(), prevMapper.get(clk));
		Value notPrevClk = comb::XorOp::create(builder, op.getLoc(), prevClk, constOne);
		Value currClk = seq::FromClockOp::create(builder, op.getLoc(), currMapper.get(clk));
		Value clockEdge = comb::AndOp::create(builder, op.getLoc(), notPrevClk, currClk);

		Value prevResult = prevMapper.get(result);
		Value currNext = currMapper.get(next);
		Value currResult = comb::MuxOp::create(builder, op.getLoc(), clockEdge, currNext, prevResult);

		currMapper.set(result, currResult);
	}
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