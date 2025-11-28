#include "circt-passes/FuncToHWModule/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseMap.h"
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

struct ArrayInfo{
  mlir::Type elementType;
  unsigned int size;
  llvm::DenseMap<unsigned int, mlir::Value> indexToValueMap;
};

// This pass restricts that:
// 1. All memrefs in the funcOp must be one-dimensional.
// 2. Arguments with the "equivfusion.direction" attribute set to "out" must have memref type.
struct FuncToHWModulePass : public circt::impl::FuncToHWModuleBase<FuncToHWModulePass> {
    using circt::impl::FuncToHWModuleBase<FuncToHWModulePass>::FuncToHWModuleBase;

    void runOnOperation() override;
private:

    mlir::Value createArrayFromArrayInfo(mlir::OpBuilder &builder, ArrayInfo &info);

    mlir::FailureOr<circt::hw::ArrayType> 
    createArrayTypeFromMemRefType(mlir::OpBuilder &builder, mlir::MemRefType memRefType);

    mlir::FailureOr<circt::hw::HWModuleOp>
    buildHWModuleOpFromFuncOp(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp, 
        llvm::DenseMap<mlir::Value, unsigned int> &funcParamInToModuleInIndexMap,
        llvm::DenseMap<mlir::Value, unsigned int> &funcParamOutToModuleOutIndexMap);

    mlir::LogicalResult
    copyOpFromFuncToHWModule(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp, circt::hw::HWModuleOp hwModuleOp,
      llvm::DenseMap<mlir::Value, unsigned int> &funcParamInToModuleInIndexMap,
      llvm::DenseMap<mlir::Value, unsigned int> &funcParamOutToModuleOutIndexMap);

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
        
        /*
        // Give up if there are any side-effecting ops in the region.
        for (auto &op : block) {
          if (!mlir::isMemoryEffectFree(&op)) {
            llvm::errs() << "Has side effects, giving up\n";
            return mlir::failure();
          }
        }
        */
    
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

mlir::Value FuncToHWModulePass::createArrayFromArrayInfo(mlir::OpBuilder &builder, ArrayInfo &info) {
  auto constantZeroOp = builder.create<circt::hw::ConstantOp>(builder.getUnknownLoc(), info.elementType, 0);
  mlir::Value zeroVal = constantZeroOp.getResult();

  llvm::SmallVector<mlir::Value> elements;
  elements.resize(info.size);
  for (unsigned int i = 0; i < info.size; i++) {
    elements[i] = info.indexToValueMap.count(i) ? info.indexToValueMap[i] : zeroVal;
  }

  circt::hw::ArrayType arrayType = circt::hw::ArrayType::get(info.elementType, info.size);
  auto createArrayOp = builder.create<circt::hw::ArrayCreateOp>(builder.getUnknownLoc(), arrayType, elements);
  return createArrayOp.getResult();
}

// Convert a memref type to hw.array type.
// Returns failure if:
// - memref has dynamic dimensions
// - memref has more than 1 dimensions
mlir::FailureOr<circt::hw::ArrayType>
FuncToHWModulePass::createArrayTypeFromMemRefType(mlir::OpBuilder &builder, mlir::MemRefType memRefType) {
  if (!memRefType.hasStaticShape()) {
    llvm::errs() << "The memref type is not a static shape.\n";
    return mlir::failure();
  }

  unsigned int rank = memRefType.getRank();
  if (rank > 1) {
    llvm::errs() << "The memref type has more than 1 dimension.\n";
    return mlir::failure();
  }

  mlir::Type elementType = memRefType.getElementType();
  unsigned int size = rank == 0 ? 1 : memRefType.getDimSize(0);

  return circt::hw::ArrayType::get(elementType, size);
}

mlir::FailureOr<circt::hw::HWModuleOp> 
FuncToHWModulePass::buildHWModuleOpFromFuncOp(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp,
  llvm::DenseMap<mlir::Value, unsigned int> &funcParamInToModuleInIndexMap,
  llvm::DenseMap<mlir::Value, unsigned int> &funcParamOutToModuleOutIndexMap) { 
    builder.setInsertionPointAfter(funcOp);
    mlir::Location loc = funcOp.getLoc();
    llvm::StringRef name = funcOp.getSymName();
    mlir::ArrayRef<mlir::Type> argumentTypes = funcOp.getArgumentTypes();
    mlir::ArrayRef<mlir::Type> resultTypes = funcOp.getResultTypes();
    std::optional<mlir::ArrayAttr> argAttrs = funcOp.getArgAttrs();

    // Get portInfo used by the builder to create hw.module
    mlir::SmallVector<circt::hw::PortInfo> inputsInfo, outputsInfo;
    for (auto [idx, argType] : llvm::enumerate(argumentTypes)) {
      std::string argName;
      std::string direction;

      if (argAttrs && idx < argAttrs->size()) {
        mlir::Attribute argAttr = (*argAttrs)[idx];
        mlir::DictionaryAttr dictAttr = llvm::dyn_cast_or_null<mlir::DictionaryAttr>(argAttr);
        if (dictAttr) {
          if (mlir::StringAttr nameAttr = dictAttr.getAs<mlir::StringAttr>("polygeist.param_name")) { 
            argName = nameAttr.getValue().str();
          }
          if (mlir::StringAttr directionAttr = dictAttr.getAs<mlir::StringAttr>("equivfusion.direction")) {
             direction = directionAttr.getValue().str();
          }
        }
      }

      bool isOut = direction == "out";
      if (argName.empty()) {
        argName = "arg_" + std::to_string(idx);
      }
      mlir::Type type = argType;
        
      if (llvm::isa<mlir::MemRefType>(type)) {
        mlir::FailureOr<circt::hw::ArrayType> typeOrFailure = createArrayTypeFromMemRefType(builder, llvm::dyn_cast<mlir::MemRefType>(type));
        if (mlir::failed(typeOrFailure)) {
          return mlir::failure();
        }
        type = *typeOrFailure;
      } else {
        if (isOut) {
          llvm::errs() << "The output argument in parameter list of '" << name.str() << "' must be a pointer.\n";
          return mlir::failure();
        }
      }
        
      circt::hw::PortInfo argInfo;
      argInfo.name = builder.getStringAttr(argName);
      argInfo.dir = isOut ? circt::hw::ModulePort::Direction::Output : circt::hw::ModulePort::Direction::Input;
      argInfo.type = type;

      // argNum represents the index of the port within input or output ports, counted separately for inputs and outputs
      if (isOut) {
        argInfo.argNum = outputsInfo.size();
        funcParamOutToModuleOutIndexMap[funcOp.getArgument(idx)] = argInfo.argNum;
        outputsInfo.push_back(argInfo);
      } else {
        argInfo.argNum = inputsInfo.size();
        funcParamInToModuleInIndexMap[funcOp.getArgument(idx)] = argInfo.argNum;
        inputsInfo.push_back(argInfo);
      }
    }

    for (auto [idx, outputType] : llvm::enumerate(resultTypes)) {
      std::string outputName = "out_" + std::to_string(outputsInfo.size());
      mlir::Type type = outputType;
      if (llvm::isa<mlir::MemRefType>(type)) {
        mlir::FailureOr<circt::hw::ArrayType> typeOrFailure = createArrayTypeFromMemRefType(builder, llvm::dyn_cast<mlir::MemRefType>(type));
        if (mlir::failed(typeOrFailure)) {
          return mlir::failure();
        }
        type = *typeOrFailure;
      }

      circt::hw::PortInfo outputInfo;
      outputInfo.name = builder.getStringAttr(outputName);
      outputInfo.dir = circt::hw::ModulePort::Direction::Output;
      outputInfo.type = type;
      outputInfo.argNum = outputsInfo.size();

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
FuncToHWModulePass::copyOpFromFuncToHWModule(mlir::OpBuilder &builder, mlir::func::FuncOp funcOp, circt::hw::HWModuleOp hwModuleOp,
    llvm::DenseMap<mlir::Value, unsigned int> &funcParamInToModuleInIndexMap,
    llvm::DenseMap<mlir::Value, unsigned int> &funcParamOutToModuleOutIndexMap) {
  // Get the body block of the hw.module
  builder.setInsertionPoint(hwModuleOp.getBodyBlock(), hwModuleOp.getBodyBlock()->begin());
  mlir::IRMapping valueMapping;
  llvm::SmallVector<mlir::Value> returnOpMappedValues;
  llvm::SmallVector<mlir::Value> outputValues;
  llvm::DenseMap<mlir::Value, ArrayInfo> memrefToArrayInfoMap;
  llvm::DenseMap<mlir::Value, mlir::Value> memrefToArrayMap;

  for (auto it : funcParamInToModuleInIndexMap) {
    valueMapping.map(it.first, hwModuleOp.getArgumentForInput(it.second));
    if (llvm::isa<mlir::MemRefType>(it.first.getType())) {
      memrefToArrayMap[it.first] = hwModuleOp.getArgumentForInput(it.second);
    }
  }

  for (auto it : funcParamOutToModuleOutIndexMap) {
    if (llvm::isa<mlir::MemRefType>(it.first.getType())) {
      ArrayInfo info;
      mlir::MemRefType memrefType = llvm::dyn_cast<mlir::MemRefType>(it.first.getType());
      info.elementType = memrefType.getElementType();
      info.size = memrefType.getRank() == 0 ? 1 : memrefType.getDimSize(0);
      memrefToArrayInfoMap[it.first] = info;
    } else {
      llvm::errs() << "The output argument in parameter list of '" << funcOp.getSymName().str() << "' must be a pointer.\n";
      return mlir::failure();
    }
  }

  // Clone operations from func to hw.module. 
  for (auto& op : llvm::make_early_inc_range(funcOp.getBody().front().getOperations())) {
    if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
      // Create hw.output operation for the return values.
      for (mlir::Value returnValue : returnOp.getOperands()) {
        mlir::Value mappedValue;

        if (llvm::isa<mlir::MemRefType>(returnValue.getType())) {
          if (funcParamInToModuleInIndexMap.count(returnValue)) { 
            mappedValue = hwModuleOp.getArgumentForInput(funcParamInToModuleInIndexMap[returnValue]);
          } else if (memrefToArrayInfoMap.count(returnValue)) {
            mappedValue = createArrayFromArrayInfo(builder, memrefToArrayInfoMap[returnValue]);
          } else {
            llvm::errs() << "The return value is used before definition.\n";
            return mlir::failure();
          }
        } else {
          mappedValue = valueMapping.lookupOrNull(returnValue);
          if (!mappedValue) { 
            llvm::errs() << "The return value " << returnValue << " is used before definition\n";
            return mlir::failure();
          }
        }

        returnOpMappedValues.push_back(mappedValue);
      }

      continue;
    }

    if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
      if (memrefToArrayMap.count(loadOp.getMemref())) {
        circt::hw::ArrayType arrayType = llvm::dyn_cast<circt::hw::ArrayType>(memrefToArrayMap[loadOp.getMemref()].getType());
        mlir::Type resultType = arrayType.getElementType();

        mlir::Value index;
        if (loadOp.getIndices().empty()) {
          auto constZeroOp = builder.create<circt::hw::ConstantOp>(loadOp.getLoc(), builder.getI32Type(), 0);
          index = constZeroOp.getResult();
        } else {
          index = valueMapping.lookupOrNull(loadOp.getIndices().front());
        }
        
        if (!index) {
          llvm::errs() << "The index of the memref.load operation is used before definition.\n";
          return mlir::failure();
        }

        auto constOp = index.getDefiningOp<circt::hw::ConstantOp>();
        if (!constOp) { 
          llvm::errs() << "The index of the memref.load operation is not a constant.\n";
          return mlir::failure();
        }

        // Adjust index bitwidth to match array size requirement
        // hw.array_get requires: index_bitwidth = ceil(log2(array_size))
        size_t arraySize = arrayType.getNumElements();
        unsigned requiredIdxWidth = (arraySize <= 1) ? 1 : llvm::Log2_64_Ceil(arraySize);
        
        mlir::Value adjustedIndex = index;
        if (mlir::IntegerType indexIntType = llvm::dyn_cast<mlir::IntegerType>(index.getType())) {
          unsigned currentWidth = indexIntType.getWidth();
          
          // If current index bitwidth doesn't match the requirement, adjust it
          if (currentWidth != requiredIdxWidth) {
            mlir::Location loc = loadOp.getLoc();
            mlir::Type targetIndexType = builder.getIntegerType(requiredIdxWidth);
            
            if (currentWidth > requiredIdxWidth) {
              // Index is too wide, truncate by extracting lower bits
              // Example: i32 -> i3, extract lower 3 bits
              adjustedIndex = builder.create<circt::comb::ExtractOp>(
                loc, targetIndexType, index, 0
              );
            } else {
              // Index is too narrow, zero-extend by padding high bits
              // Example: i2 -> i3, pad 1 bit of zero at high position
              mlir::Value zeroPad = builder.create<circt::hw::ConstantOp>(
                loc,
                builder.getIntegerType(requiredIdxWidth - currentWidth),
                0
              );
              // Concatenate: {high zeros, original index} to form new index
              adjustedIndex = builder.create<circt::comb::ConcatOp>(
                loc, zeroPad, index
              );
            }
          }
        } else {
          loadOp.emitError("Index must be integer type");
          return mlir::failure();
        }

        auto resultOp = builder.create<circt::hw::ArrayGetOp>(loadOp.getLoc(), resultType, memrefToArrayMap[loadOp.getMemref()], adjustedIndex);
        valueMapping.map(loadOp.getResult(), resultOp.getResult());
      } else if (funcParamOutToModuleOutIndexMap.count(loadOp.getMemref())) {
        llvm::errs() << "The memref which is an output parameter is used by memref.load.\n";
        return mlir::failure();
      } else {
        llvm::errs() << "The memref is used before definition.\n";
        return mlir::failure();
      }

      continue;
    }

    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
      if (memrefToArrayInfoMap.count(storeOp.getMemref())) {
        ArrayInfo &info = memrefToArrayInfoMap[storeOp.getMemref()];
        mlir::Value value = valueMapping.lookupOrNull(storeOp.getValue());
        mlir::Value indexVal;
        llvm::APInt index;
        unsigned int idx;

        if (storeOp.getIndices().empty()) {
          auto constZeroOp = builder.create<circt::hw::ConstantOp>(storeOp.getLoc(), builder.getI32Type(), 0);
          indexVal = constZeroOp.getResult();
        } else {
          indexVal = valueMapping.lookupOrNull(storeOp.getIndices().front());
        }

        if (!value) {
          llvm::errs() << "The value of the memref.store operation is used before definition.\n";
          return mlir::failure();
        }

        if (!indexVal) {
          llvm::errs() << "The index of the memref.store operation is used before definition.\n";
          return mlir::failure();
        }

        if (auto constOp = indexVal.getDefiningOp<circt::hw::ConstantOp>()) {
          index = constOp.getValue();
          idx = index.getZExtValue();
          info.indexToValueMap[idx] = value;
        } else {
          llvm::errs() << "The index of the memref.store operation is not a constant.\n";
          return mlir::failure();
        }
      } else if (funcParamInToModuleInIndexMap.count(storeOp.getMemref())) {
        llvm::errs() << "The memref which is an input parameter is used by memref.store.\n";
        return mlir::failure();
      } else {
        llvm::errs() << "The memref is used before definition.\n";
        return mlir::failure();
      }

      continue;
    }

    if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) { 
      mlir::Value resultValue = allocOp.getResult();
      mlir::MemRefType memRefType = llvm::dyn_cast<mlir::MemRefType>(resultValue.getType());
      
      mlir::FailureOr<circt::hw::ArrayType> arrayType = createArrayTypeFromMemRefType(builder, memRefType);
      if (mlir::failed(arrayType)) {
        return mlir::failure();
      }

      ArrayInfo arrayInfo;
      arrayInfo.size = (*arrayType).getNumElements();
      arrayInfo.elementType = (*arrayType).getElementType();
      memrefToArrayInfoMap[resultValue] = arrayInfo;
      
      mlir::Value arrayValue = createArrayFromArrayInfo(builder, arrayInfo);
      memrefToArrayMap[resultValue] = arrayValue;

      valueMapping.map(resultValue, arrayValue);

      continue;
    }

    if (auto allocaOp = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
      mlir::Value resultValue = allocaOp.getResult();
      mlir::MemRefType memRefType = llvm::dyn_cast<mlir::MemRefType>(resultValue.getType());

      mlir::FailureOr<circt::hw::ArrayType> arrayType = createArrayTypeFromMemRefType(builder, memRefType);
      if (mlir::failed(arrayType)) {
        return mlir::failure();
      }
      
      ArrayInfo arrayInfo;
      arrayInfo.size = (*arrayType).getNumElements();
      arrayInfo.elementType = (*arrayType).getElementType();
      memrefToArrayInfoMap[resultValue] = arrayInfo;

      mlir::Value arrayValue = createArrayFromArrayInfo(builder, arrayInfo);
      memrefToArrayMap[resultValue] = arrayValue;

      valueMapping.map(resultValue, arrayValue);

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

  outputValues.resize(funcParamOutToModuleOutIndexMap.size());
  for (auto it : funcParamOutToModuleOutIndexMap) {
    outputValues[it.second] = createArrayFromArrayInfo(builder, memrefToArrayInfoMap[it.first]);
  }

  for (auto value : returnOpMappedValues) {
    outputValues.push_back(value);
  }
  builder.setInsertionPoint(hwModuleOp.getBodyBlock(), hwModuleOp.getBodyBlock()->end());
  builder.create<circt::hw::OutputOp>(hwModuleOp.getLoc(), outputValues);

  return mlir::success();
}

mlir::LogicalResult FuncToHWModulePass::convertFuncToHWModule(mlir::func::FuncOp funcOp) {
    mlir::OpBuilder builder(funcOp.getContext());
    llvm::DenseMap<mlir::Value, unsigned int> funcParamInToModuleInIndexMap;
    llvm::DenseMap<mlir::Value, unsigned int> funcParamOutToModuleOutIndexMap;

    auto hwModuleOpOrFailure = buildHWModuleOpFromFuncOp(builder, funcOp, funcParamInToModuleInIndexMap, funcParamOutToModuleOutIndexMap);
    if (mlir::failed(hwModuleOpOrFailure)) {
      return mlir::failure();
    }

    if (mlir::failed(copyOpFromFuncToHWModule(builder, funcOp, *hwModuleOpOrFailure, 
        funcParamInToModuleInIndexMap, funcParamOutToModuleOutIndexMap))) {
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
