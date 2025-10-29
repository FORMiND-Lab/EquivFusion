#include "circt-passes/FuncToHWModule/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Casting.h"

namespace circt {
#define GEN_PASS_DEF_FUNCTOHWMODULE
#include "circt-passes/FuncToHWModule/Passes.h.inc"
} // namespace circt
 

namespace {
/// A helper struct that tracks a boolean condition as either a constant false,
/// constant true, or an SSA value.
struct Condition {
    Condition() {}
    Condition(mlir::Value value) : pair(value, 0) {
      if (value) {
        if (mlir::matchPattern(value, mlir::m_One()))
          *this = Condition(true);
        if (mlir::matchPattern(value, mlir::m_Zero()))
          *this = Condition(false);
      }
    }
    Condition(bool konst) : pair(nullptr, konst ? 1 : 2) {}
  
    explicit operator bool() const {
      return pair.getPointer() != nullptr || pair.getInt() != 0;
    }
  
    bool isTrue() const { return !pair.getPointer() && pair.getInt() == 1; }
    bool isFalse() const { return !pair.getPointer() && pair.getInt() == 2; }
    mlir::Value getValue() const { return pair.getPointer(); }
  
    /// Turn this condition into an SSA value, creating an `hw.constant` if the
    /// condition is a constant.
    mlir::Value materialize(mlir::OpBuilder &builder, mlir::Location loc) const {
      if (isTrue())
        return circt::hw::ConstantOp::create(builder, loc, llvm::APInt(1, 1));
      if (isFalse())
        return circt::hw::ConstantOp::create(builder, loc, llvm::APInt(1, 0));
      return pair.getPointer();
    }
  
    Condition orWith(Condition other, mlir::OpBuilder &builder) const {
      if (isTrue() || other.isTrue())
        return true;
      if (isFalse())
        return other;
      if (other.isFalse())
        return *this;
      return builder.createOrFold<circt::comb::OrOp>(getValue().getLoc(), getValue(),
                                              other.getValue());
    }
  
    Condition andWith(Condition other, mlir::OpBuilder &builder) const {
      if (isFalse() || other.isFalse())
        return false;
      if (isTrue())
        return other;
      if (other.isTrue())
        return *this;
      return builder.createOrFold<circt::comb::AndOp>(getValue().getLoc(), getValue(),
                                               other.getValue());
    }
  
    Condition inverted(mlir::OpBuilder &builder) const {
      if (isTrue())
        return false;
      if (isFalse())
        return true;
      return circt::comb::createOrFoldNot(getValue().getLoc(), getValue(), builder);
    }
  
  private:
    llvm::PointerIntPair<mlir::Value, 2> pair;
  };



struct CFRemover {
  CFRemover(mlir::Region &region) : region(region) {}
  mlir::LogicalResult run();

  /// The region within which we are removing control flow.
  mlir::Region &region;
  /// The blocks in the region, sorted such that a block's predecessors appear
  /// in the list before the block itself.
  llvm::SmallVector<mlir::Block *> sortedBlocks;
  /// The dominance information for the region.
  mlir::DominanceInfo domInfo;
};


struct FuncToHWModulePass : public circt::impl::FuncToHWModuleBase<FuncToHWModulePass> {
    using circt::impl::FuncToHWModuleBase<FuncToHWModulePass>::FuncToHWModuleBase;

    void runOnOperation() override;
private:

    circt::hw::HWModuleOp
    buildHWModuleOPFromFuncOP(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp);

    mlir::LogicalResult
    copyOpFromFuncToHWModule(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp, circt::hw::HWModuleOp hwModuleOp);

    mlir::LogicalResult convertFuncToHWModule(mlir::func::FuncOp funcOp);

    mlir::LogicalResult
    removeControlFlowFromFuncOp(mlir::func::FuncOp funcOp);
};

} // namespace

static Condition getBranchDecisionsFromDominatorToTarget(
    mlir::OpBuilder &builder, mlir::Block *dominator, mlir::Block *target,
    llvm::SmallDenseMap<std::pair<mlir::Block *, mlir::Block *>, Condition> &decisions) {
  if (auto decision = decisions.lookup({dominator, target}))
    return decision;

  llvm::SmallPtrSet<mlir::Block *, 8> visitedBlocks;
  visitedBlocks.insert(dominator); // stop at the dominator
  if (auto &decision = decisions[{dominator, dominator}]; !decision)
    decision = Condition(true);

  // Traverse the blocks in inverse post order. This ensures that we are
  // visiting all of a block's predecessors before we visit the block itself.
  // This allows us to first compute the decision leading control flow to each
  // of the predecessors, such that the current block can then just combine the
  // predecessor decisions.
  for (auto *block : llvm::inverse_post_order_ext(target, visitedBlocks)) {
    auto merged = Condition(false);
    for (auto *pred : block->getPredecessors()) {
      auto predDecision = decisions.lookup({dominator, pred});
      assert(predDecision);
      if (pred->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = llvm::cast<mlir::cf::CondBranchOp>(pred->getTerminator());
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          merged = merged.orWith(predDecision, builder);
        } else {
          auto cond = Condition(condBr.getCondition());
          if (condBr.getFalseDest() == block)
            cond = cond.inverted(builder);
          merged = merged.orWith(cond.andWith(predDecision, builder), builder);
        }
      } else {
        merged = merged.orWith(predDecision, builder);
      }
    }
    assert(merged);
    decisions.insert({{dominator, block}, merged});
  }

  return decisions.lookup({dominator, target});
}

mlir::LogicalResult
CFRemover::run() {
    // Establish a topological order of the blocks in the region. Give up if we
    // detect a control flow cycle. Also take note of all ReturnOp, such that we
    // can combine them into a single return block later.

    llvm::SmallVector<mlir::func::ReturnOp, 2> returnOps;
    llvm::SmallPtrSet<mlir::Block *, 8> visitedBlocks, ipoSet;

    for (auto &block : region) {
        for (auto *ipoBlock : llvm::inverse_post_order_ext(&block, ipoSet)) {
          if (!llvm::all_of(ipoBlock->getPredecessors(), [&](auto *pred) {
                return visitedBlocks.contains(pred);
              })) {
            llvm::errs() << "Loop detected, giving up\n";
            return mlir::failure();
          }
          visitedBlocks.insert(ipoBlock);
          sortedBlocks.push_back(ipoBlock);
        }
    
        // Give up if there are any side-effecting ops in the region.
        for (auto &op : block) {
          if (!mlir::isMemoryEffectFree(&op)) {
            llvm::errs() << "Has side effects, giving up\n";
            return mlir::failure();
          }
        }
    
        // Check that we know what to do with all terminators.
        if (!llvm::isa<mlir::func::ReturnOp, mlir::cf::BranchOp, mlir::cf::CondBranchOp>(block.getTerminator())) {
          llvm::errs() << "Has unsupported terminator " << block.getTerminator()->getName() << ", giving up\n";
            return mlir::failure();
        }
    
        // Keep track of return ops.
        if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator()))
            returnOps.push_back(returnOp);
    }

    // If there are multiple return ops, factor them out into a single return block.
    auto returnOp = returnOps[0];
    if (returnOps.size() > 1) {
        mlir::OpBuilder builder(region.getContext());
        llvm::SmallVector<mlir::Location> locs(returnOps[0].getNumOperands(), region.getLoc());
        auto *returnBlock = builder.createBlock(&region, region.end(),
                                               returnOps[0].getOperandTypes(), locs);
        sortedBlocks.push_back(returnBlock);
        returnOp =
            mlir::func::ReturnOp::create(builder, region.getLoc(), returnBlock->getArguments());
        for (auto returnOp : returnOps) {
          builder.setInsertionPoint(returnOp);
          mlir::cf::BranchOp::create(builder, returnOp.getLoc(), returnBlock,
                               returnOp.getOperands());
          returnOp.erase();
        }
    }

    // Compute the dominance info for this region.
    domInfo = mlir::DominanceInfo(region.getParentOp());

    // Move operations into the entry block, replacing block arguments with
    // multiplexers as we go. The block order guarantees that we visit a block's
    // predecessors before we visit the block itself.

    llvm::SmallDenseMap<std::pair<mlir::Block *, mlir::Block *>, Condition> decisionCache;
    auto *entryBlock = sortedBlocks.front();

    for (auto *block : sortedBlocks) {
        if (!domInfo.isReachableFromEntry(block))
            continue;

        // Find the nearest common dominator block of all predecessors. Any block
        // arguments reaching the current block will only depend on control flow
        // decisions between this dominator block and the current block.
        auto *domBlock = block;
        for (auto *pred : block->getPredecessors())
            if (domInfo.isReachableFromEntry(pred))
                domBlock = domInfo.findNearestCommonDominator(domBlock, pred);
        
        // Convert the block arguments into multiplexers.
        mlir::OpBuilder builder(entryBlock->getTerminator());
        mlir::SmallVector<mlir::Value> mergedArgs;
        mlir::SmallPtrSet<mlir::Block *, 4> seenPreds;

        for (auto *pred : block->getPredecessors()) {
            // A block may be listed multiple times in the predecessors.
            if (!seenPreds.insert(pred).second)
                continue;

            // Only consider values coming from reachable predecessors.
            if (!domInfo.isReachableFromEntry(pred))
                continue;
            
            // Helper function to create a multiplexer between the current
            // `mergedArgs` and a new set of `args`, where the new args are picked if
            // `cond` is true and control flows from `domBlock` to `pred`. The
            // condition may be null, in which case the mux will directly use the
            // branch decisions that lead from `domBlock` to `pred`.

            auto mergeArgs = [&](mlir::ValueRange args, Condition cond, bool invCond) {
                if (mergedArgs.empty()) {
                    mergedArgs = args;
                    return;
                }
                auto decision = getBranchDecisionsFromDominatorToTarget(
                    builder, domBlock, pred, decisionCache);
                if (cond) {
                    if (invCond)
                        cond = cond.inverted(builder);
                    decision = decision.andWith(cond, builder);
                }
                for (auto [mergedArg, arg] : llvm::zip(mergedArgs, args)) {
                  if (decision.isTrue())
                      mergedArg = arg;
                  else if (decision.isFalse())
                      continue;
                  else
                    mergedArg = builder.createOrFold<circt::comb::MuxOp>(
                        arg.getLoc(), decision.materialize(builder, arg.getLoc()), arg,
                        mergedArg);
                }
            };

            // Handle the different terminators that we support.
            if (auto condBrOp = llvm::dyn_cast<mlir::cf::CondBranchOp>(pred->getTerminator())) {
                if (condBrOp.getTrueDest() == condBrOp.getFalseDest()) {
                  // Both destinations lead to the current block. Insert a mux to
                  // collapse the destination operands and then treat this as an
                  // unconditional branch to the current block.
                  llvm::SmallVector<mlir::Value> mergedOperands;
                  mergedOperands.reserve(block->getNumArguments());
                  for (auto [trueArg, falseArg] :
                       llvm::zip(condBrOp.getTrueDestOperands(),
                                 condBrOp.getFalseDestOperands())) {
                    mergedOperands.push_back(builder.createOrFold<circt::comb::MuxOp>(
                        trueArg.getLoc(), condBrOp.getCondition(), trueArg, falseArg));
                  }
                  mergeArgs(mergedOperands, mlir::Value{}, false);
                } else if (condBrOp.getTrueDest() == block) {
                  // The branch leads to the current block if the condition is true.
                  mergeArgs(condBrOp.getTrueDestOperands(), condBrOp.getCondition(),
                            false);
                } else {
                  // The branch leads to the current block if the condition is false.
                  mergeArgs(condBrOp.getFalseDestOperands(), condBrOp.getCondition(),
                            true);
                }
            } else {
                auto brOp = llvm::cast<mlir::cf::BranchOp>(pred->getTerminator());
                mergeArgs(brOp.getDestOperands(), mlir::Value{}, false);
            }
        } 

        for (auto [blockArg, mergedArg] :
               llvm::zip(block->getArguments(), mergedArgs))
            blockArg.replaceAllUsesWith(mergedArg);
   
       // Move all ops except for the terminator into the entry block.
       if (block != entryBlock)
         entryBlock->getOperations().splice(--entryBlock->end(),
                                            block->getOperations(), block->begin(),
                                            --block->end());

    }

    // Move the return op into the entry block, replacing the original terminator.
    if (returnOp != entryBlock->getTerminator()) {
        returnOp->moveBefore(entryBlock->getTerminator());
        entryBlock->getTerminator()->erase();
    }

    // Remove all blocks except for the entry block. We first clear all operations
    // in the blocks such that the blocks have no more uses in branch ops. Then we
    // remove the blocks themselves in a second pass.
     for (auto *block : sortedBlocks)
        if (block != entryBlock)
            block->clear();
    for (auto *block : sortedBlocks)
        if (block != entryBlock)
            block->erase();

    return mlir::success();
}

circt::hw::HWModuleOp 
FuncToHWModulePass::buildHWModuleOPFromFuncOP(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp) { 
    builder.setInsertionPointAfter(funcOp);
    mlir::Location loc = funcOp.getLoc();
    llvm::StringRef name = funcOp.getSymName();
    mlir::ArrayRef<mlir::Type> argumentTypes = funcOp.getArgumentTypes();
    mlir::ArrayRef<mlir::Type> resultTypes = funcOp.getResultTypes();
    std::optional<mlir::ArrayAttr> argAttrs = funcOp.getArgAttrs();

    // Get portInfo used by the builder to create hw.module
    mlir::SmallVector<circt::hw::PortInfo> inputsInfo, outputsInfo;
    for (auto [idx, inputType] : llvm::enumerate(argumentTypes)) {
        // get the name of input
        std::string inputName;
        if (argAttrs && idx < argAttrs->size()) {
            mlir::Attribute argAttr = (*argAttrs)[idx];
            mlir::DictionaryAttr dictAttr = llvm::dyn_cast_or_null<mlir::DictionaryAttr>(argAttr);
            if (dictAttr) {
                if (mlir::StringAttr nameAttr = dictAttr.getAs<mlir::StringAttr>("polygeist.param_name")) { 
                    inputName = nameAttr.getValue().str();
                }
            }
        }

        if (inputName.empty()) {
            inputName = "in_" + std::to_string(idx);
        }

        circt::hw::PortInfo inputInfo;
        inputInfo.name = builder.getStringAttr(inputName);
        inputInfo.dir = circt::hw::ModulePort::Direction::Input;
        inputInfo.type = inputType;
        inputInfo.argNum = idx;

        inputsInfo.push_back(inputInfo);
    }

    for (auto [idx, outputType] : llvm::enumerate(resultTypes)) {
        std::string outputName = "out_" + std::to_string(idx);
        circt::hw::PortInfo outputInfo;
        outputInfo.name = builder.getStringAttr(outputName);
        outputInfo.dir = circt::hw::ModulePort::Direction::Output;
        outputInfo.type = outputType;
        outputInfo.argNum = idx;

        outputsInfo.push_back(outputInfo);
    }

    circt::hw::ModulePortInfo portInfo(inputsInfo, outputsInfo);
    
    // Create the hw.module operation
    auto hwModule = circt::hw::HWModuleOp::create(
        builder,                           // OpBuilder
        loc,                              // Location
        builder.getStringAttr(name),      // Module name
        portInfo,                         // Port information
        {},                               // Parameters (empty)
        {},                               // Attributes (empty)
        {},                               // Comment (empty)
        false                              // shouldEnsureTerminator
    );

    return hwModule;
}

mlir::LogicalResult
FuncToHWModulePass::copyOpFromFuncToHWModule(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp, circt::hw::HWModuleOp hwModuleOp) {
    // Get the bofy block of the hw.module
    builder.setInsertionPoint(hwModuleOp.getBodyBlock(), hwModuleOp.getBodyBlock()->begin());

    mlir::Block &funcOpBody = funcOp.getBody().front();
    size_t moduleInPortNum = hwModuleOp.getNumInputPorts();
    llvm::SmallVector<mlir::Value> moduleInPortValues;

    // Get the values of the input ports of the hw.module
    for (size_t i = 0; i < moduleInPortNum; ++i) {
        moduleInPortValues.push_back(hwModuleOp.getArgumentForInput(i));
    }

    mlir::IRMapping valueMapping;
    
    // Ensure the number of arguments in the function body is equal to the number of input ports in the hw.module//
    if (funcOpBody.getNumArguments() != moduleInPortNum) { 
        llvm::errs() << "The number of arguments in the function body is not equal to the number of input ports in the hw.module\n";
        return mlir::failure();
    } 

    for (auto [idx, funcArg] : llvm::enumerate(funcOpBody.getArguments())) {
        mlir::BlockArgument hwInArg = hwModuleOp.getArgumentForInput(idx);
        valueMapping.map(funcArg, hwInArg);
    }

    // Clone operations from func to hw.module. 
    for (auto& op : llvm::make_early_inc_range(funcOpBody.getOperations())) {
        if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
            // Create hw.output operation for the return values.
            llvm::SmallVector<mlir::Value> outputValues;
            for (mlir::Value returnValue : returnOp.getOperands()) {
                auto mappedValue = valueMapping.lookupOrNull(returnValue);
                if (!mappedValue) { 
                    llvm::errs() << "The return value " << returnValue << " is used before definition\n";
                    return mlir::failure();
                }

                outputValues.push_back(mappedValue);
            }

            builder.create<circt::hw::OutputOp>(op.getLoc(), outputValues);
            continue;
        }

        mlir::Operation* clonedOp = builder.clone(op, valueMapping);
        if (!clonedOp) {
          llvm::errs() << "Failed to clone operation " << op << "\n";
          return mlir::failure();
        }
         
        for (auto [originalResult, clonedResult] : 
            llvm::zip(op.getResults(), clonedOp->getResults())) {
            valueMapping.map(originalResult, clonedResult);
        }
    }

    
    return mlir::success();
}

mlir::LogicalResult FuncToHWModulePass::convertFuncToHWModule(mlir::func::FuncOp funcOp) {
    mlir::OpBuilder builder(funcOp.getContext());
    circt::hw::HWModuleOp hwModuleOp = buildHWModuleOPFromFuncOP(builder, funcOp);
    
    if (failed(copyOpFromFuncToHWModule(builder, funcOp, hwModuleOp))) {
        return mlir::failure();
    }
    

    return mlir::success();
}

mlir::LogicalResult FuncToHWModulePass::removeControlFlowFromFuncOp(mlir::func::FuncOp funcOp) {
    CFRemover remover(funcOp.getBody());
    return remover.run();
}

void FuncToHWModulePass::runOnOperation() {
    for (auto op : llvm::make_early_inc_range(getOperation().getOps<mlir::func::FuncOp>())) {
        if (mlir::failed(removeControlFlowFromFuncOp(op))) {
            return signalPassFailure();
        }

        if (mlir::failed(convertFuncToHWModule(op))) {
            return signalPassFailure();
        }

        op.erase();
    }
}
