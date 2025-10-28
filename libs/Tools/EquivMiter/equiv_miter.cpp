#include "infrastructure/log/log.h"
#include "libs/Tools/EquivMiter/equiv_miter.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"             // parseSourceFile                  MLIRParser

#include "circt/Dialect/OM/OMPasses.h"      // createStripOMPass                CIRCTOMTransforms
#include "circt/Dialect/Emit/EmitPasses.h"  // createStripEmitPass              CIRCTEmitTransforms
#include "circt/Dialect/HW/HWPasses.h"      // createFlattenModules             CIRCTHWTransforms
#include "circt-passes/Miter/Passes.h"      // createEquivFusionMiter           EquivFusionPassEquivMiter

using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START

// Move all operations in `src` to `dest`. Rename all symbols in `src` to avoid conflict.
FailureOr<StringAttr> EquivMiterTool::mergeModules(ModuleOp dest, ModuleOp src, StringAttr name) {
    SymbolTable destTable(dest), srcTable(src);
    StringAttr newName = {};
    for (auto &op : src.getOps()) {
        if (SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op)) {
            auto oldSymbol = symbol.getNameAttr();
            auto result = srcTable.renameToUnique(&op, {&destTable});
            if (failed(result))
                return src->emitError() << "failed to rename symbol " << oldSymbol;

            if (oldSymbol == name) {
                assert(!newName && "symbol must be unique");
                newName = *result;
            }
        }
    }

    if (!newName)
        return src->emitError()
               << "module " << name << " was not found in the second module";

    dest.getBody()->getOperations().splice(dest.getBody()->begin(),
                                           src.getBody()->getOperations());
    return newName;
}

// Parse one or two MLIR modules and merge it into a single module.
FailureOr<OwningOpRef<ModuleOp>> EquivMiterTool::parseAndMergeModules(MLIRContext &context) {
    auto module = parseSourceFile<ModuleOp>(inputFilenames_[0], &context);
    if (!module)
        return failure();

    if (inputFilenames_.size() == 2) {
        auto moduleOpt = parseSourceFile<ModuleOp>(inputFilenames_[1], &context);
        if (!moduleOpt)
            return failure();
        auto result = mergeModules(module.get(), moduleOpt.get(),
                                   StringAttr::get(&context, secondModuleName_));
        if (failed(result))
            return failure();
        
        secondModuleName_ = result->getValue().str();
    }

    return module;
}

/// The entry point for the `equiv_miter` tool
bool EquivMiterTool::run(MLIRContext& context, OwningOpRef<ModuleOp> &outputModule) {
    // Parse and merge modules
    auto parsedModule = parseAndMergeModules(context);
    if (failed(parsedModule)) {
        return -1;
    }
    OwningOpRef<ModuleOp> module = std::move(parsedModule.value());
    
    PassManager pm(&context);
    
    EquivFusionMiterOptions opts = {firstModuleName_, secondModuleName_, miterMode_};

    switch (miterMode_) {
    case EquivFusionMiter::MiterModeEnum::SMTLIB:
        pm.addPass(om::createStripOMPass());
        pm.addPass(emit::createStripEmitPass());
        pm.addPass(hw::createFlattenModules());
        pm.addPass(createEquivFusionMiter(opts));
        break;
    case EquivFusionMiter::MiterModeEnum::AIGER:
        pm.addPass(createEquivFusionMiter(opts));
        break;
    case EquivFusionMiter::MiterModeEnum::BTOR2:
        pm.addPass(createEquivFusionMiter(opts));
        break;
    }

    if (failed(pm.run(module.get())))
        return false;

    if (verbose_) {
        module.get().dump();
    }

    outputModule = std::move(module);
    return true;
}

bool EquivMiterTool::initOptions(const std::vector<std::string> &args) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "--c1" && idx + 1 < args.size()) {
            firstModuleName_ = args[++idx];
        } else if (arg == "--c2" && idx + 1 < args.size()) {
            secondModuleName_ = args[++idx];
        } else if (arg == "--mitermode" && idx + 1 < args.size()) {
            auto val = args[++idx];
            if (val == "aiger") {
                miterMode_ = EquivFusionMiter::MiterModeEnum::AIGER;
            } else if (val == "btor2") {
                miterMode_ = EquivFusionMiter::MiterModeEnum::BTOR2;
            } else if (val == "smtlib") {
                miterMode_ = EquivFusionMiter::MiterModeEnum::SMTLIB;
            } else {
                log("Wrong option value of --mitermode.\n");
                return false;
            }
        } else if (arg == "--verbose") {
            verbose_ = true;
        } else {
            inputFilenames_.push_back(arg);
        }
    }    

    if (firstModuleName_.empty() || secondModuleName_.empty()) {
        log("Both --c1 and --c2 must be specified.\n");
        return false;
    }

    if (inputFilenames_.empty() || inputFilenames_.size() > 2) {
        log("Must provide 1 or 2 input files.\n");
        return false;
    }

    return true;
}

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
