#include "circt-passes/TemporalUnroll/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/IRMapping.h"

#include <iostream>

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
	/// Main unrolling implementation
    LogicalResult unrollModule(OpBuilder &builder, hw::HWModuleOp module);

	hw::HWModuleOp createUnrollModule(OpBuilder &builder, hw::HWModuleOp module);

	void initializeMapping(hw::HWModuleOp module, hw::HWModuleOp newModule, unsigned step);
	void unrollFirRegOp(OpBuilder &builder, seq::FirRegOp op, unsigned step);
	LogicalResult creatUnrollModuleBody(OpBuilder &builder, hw::HWModuleOp module, hw::HWModuleOp newModule);

private:
	IRMapping prevMapping_;
	IRMapping currMapping_;
};
} // namespace

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

void EquivFusionTemporalUnrollPass::initializeMapping(hw::HWModuleOp module, hw::HWModuleOp newModule, unsigned step) {
	prevMapping_ = std::move(currMapping_);
	currMapping_.clear();

	Block *oldBody = module.getBodyBlock();
	Block *newBody = newModule.getBodyBlock();

	auto oldArgumentsSize = oldBody->getArguments().size();
	for (unsigned argIdx = 0; argIdx < oldArgumentsSize; ++argIdx) {
		Value oldArgValue = oldBody->getArgument(argIdx);
		unsigned newIndex = argIdx + step * oldArgumentsSize;
		Value newArgValue = newBody->getArgument(newIndex);
		currMapping_.map(oldArgValue, newArgValue);
 	}
}

void EquivFusionTemporalUnrollPass::unrollFirRegOp(OpBuilder &builder, seq::FirRegOp op, unsigned step) {
	// TODO (taomengxia): reset value
	auto next = op.getNext();
	auto clk = op.getClk();
	auto result = op.getResult();

	if (step == 0) {
		// TODO(taomengxia): initial value
		// currResult = currNext
		currMapping_.map(result, currMapping_.lookupOrNull(next));
	} else {
		// currResult = (!prevClk && currClk) ? currNext : prevResult
		Value constOne = hw::ConstantOp::create(builder, op.getLoc(), builder.getI1Type(), 1);
		Value prevClk = seq::FromClockOp::create(builder, op.getLoc(), prevMapping_.lookupOrNull(clk));
		Value notPrevClk = comb::XorOp::create(builder, op.getLoc(), prevClk, constOne);
		Value currClk = seq::FromClockOp::create(builder, op.getLoc(), currMapping_.lookupOrNull(clk));
		Value clockEdge = comb::AndOp::create(builder, op.getLoc(), notPrevClk, currClk);

		Value prevResult = prevMapping_.lookupOrNull(result);
		Value currNext = currMapping_.lookupOrNull(next);
		Value currResult = comb::MuxOp::create(builder, op.getLoc(), clockEdge, currNext, prevResult);

		currMapping_.map(result, currResult);
	}
}

LogicalResult EquivFusionTemporalUnrollPass::creatUnrollModuleBody(OpBuilder &builder, hw::HWModuleOp module,
																	hw::HWModuleOp newModule) {
	Block *newBlock = newModule.getBodyBlock();
	builder.setInsertionPointToEnd(newBlock);

	SmallVector<Value> newOutputValues;;
	Block *oldBlock = module.getBodyBlock();
	for (unsigned step = 0; step < timeSteps; ++step) {
		initializeMapping(module, newModule, step);

		for (auto &op : oldBlock->getOperations()) {
			if (auto outputOp = dyn_cast<hw::OutputOp>(op)) {
				// Handle output operaton by collecting output values
				for (auto outputVal : outputOp.getOperands()) {
					Value mappedVal = currMapping_.lookupOrNull(outputVal);
					newOutputValues.push_back(mappedVal);
				}
			} else if (auto firregOp = dyn_cast<seq::FirRegOp>(op)) {
				unrollFirRegOp(builder, firregOp, step);
			} else {
				// Clone operation with mapping
				builder.clone(op, currMapping_);
			}
		}
	}

	// Create final output operation
	builder.setInsertionPointToEnd(newBlock);
	builder.create<hw::OutputOp>(newModule.getLoc(), newOutputValues);
	return success();
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