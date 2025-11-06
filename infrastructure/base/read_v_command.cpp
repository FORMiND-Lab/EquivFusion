#include "infrastructure/base/command.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/utils/log/log.h"

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"

#include "circt/Conversion/ImportVerilog.h"
#include "circt/Conversion/MooreToCore.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"

#include "circt-passes/ConvertCaseToLogicalComparsion/Passes.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/Timing.h"


XUANSONG_NAMESPACE_HEADER_START

namespace {

struct ReadVCommandOptions {
    bool spec = false;
    bool impl = false;
    bool allowUseBeforeDeclare = false;
    bool ignoreUnknownModules = false;

    std::vector<std::string> inputFiles;
    std::vector<std::string> includeDirs;
    std::vector<std::string> includeSystemDirs;
    std::vector<std::string> libDirs;
    std::vector<std::string> libExts;
    std::vector<std::string> excludeExts;
    std::vector<std::string> defines;
    std::vector<std::string> undefines;
    std::vector<std::string> paramOverrides;
    std::vector<std::string> libraryFiles;

    std::string topModuleName = "";
    std::string timeScale = "";
};

} // namespace


struct ReadVCommand : public Command {
    ReadVCommand() : Command("read_v", "read rtl files") {}
    ~ReadVCommand() = default;

    ReadVCommand(const ReadVCommand &) = delete;
    ReadVCommand &operator=(const ReadVCommand &) = delete;
    ReadVCommand(ReadVCommand &&) = delete;
    ReadVCommand &operator=(ReadVCommand &&) = delete;

    // Optimize and simplify the Moore dialect IR.
    void populateMooreTransforms(mlir::PassManager &pm);

    // Convert Moore dialect IR into core dialect IR.
    void populateMooreToCoreLowering(mlir::PassManager &pm);

    // Convert LLHD dialect IR into core dialect IR
    void populateLLHDLowering(mlir::PassManager &pm);

    bool startWith(const std::string& str, const std::string& prefix);
    bool parseOptions(const std::vector<std::string>& args, ReadVCommandOptions& opts);
    llvm::LogicalResult executeRTLFlow(const ReadVCommandOptions& opts);

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        ReadVCommandOptions opts;
        if (!parseOptions(args, opts)) {
            return;
        }

        if (llvm::failed(executeRTLFlow(opts))) {
            logError("Command 'read_v' Failed!\n");
            return;
        }

/*
        // Debug Code
        if (opts.spec) {
            EquivFusionManager::getInstance()->getSpecModuleOp().print(llvm::outs());
        } else {
            EquivFusionManager::getInstance()->getImplModuleOp().print(llvm::outs());
        }
*/


        return;
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    read_v [options] <--spec | --impl> <inputFiles>\n");
        log("   OPTIONS:\n");
        log("       --top <topModuleName> ---------------------- Specify top module name\n");
        log("       --spec ------------------------------------- Design is specification\n");
        log("       --impl ------------------------------------- Design is implementation\n");
        log("       --I<dir> | --I <dir> ----------------------- Additional include search paths\n");
        log("       --isystem <dir> ---------------------------- Additional system include search paths\n");
        log("       --y<dir> | --y <dir> ----------------------- Library search paths, which will be searched for missing modules\n");
        log("       --Y<ext> | --Y <ext> ----------------------- Additional library file extensions to search\n");
        log("       --exclude-ext <ext> ------------------------ Exclude provided source files with these extensions\n");
        log("       --D<macro>=<value> | --D <macro>=<value> --- Define <macro> to <value> (or 1 if <value> omitted) in all source files\n");
        log("       --U<macro> | --U <macro> ------------------- Undefine macro name at the start of all source files\n");
        log("       --timescale <base>/<precision> ------------- Default time scale to use for design elements that don't specify one explicitly\n");
        log("       --allow-use-before-declare ----------------- Don't issue an error for use of names before their declarations\n");
        log("       --ignore-unknown-modules ------------------- Don't issue an error for instantiations of unknown modules, interface, and programs\n");
        log("       --G<name>=<value> | --G <name>=<value> ----- One or more parameter overrides to apply when instantiating top-level module\n");
        log("       --l<filename> | --l <filename> ------------- Library files, which are separate compilation units where modules are not automatically instantiated\n");
        log("\n");
    }

} readVCommand;

void ReadVCommand::populateMooreTransforms(mlir::PassManager &pm) {
    {
        // Perform an initial cleanup and preprocessing across all
        // modules/functions.
        auto &anyPM = pm.nestAny();
        anyPM.addPass(mlir::createCSEPass());
        anyPM.addPass(mlir::createCanonicalizerPass());
    }
    
    // Remove unused symbols.
    pm.addPass(mlir::createSymbolDCEPass());
    
    {
        // Perform module-specific transformations.
        auto &modulePM = pm.nest<circt::moore::SVModuleOp>();
        modulePM.addPass(circt::moore::createLowerConcatRefPass());
        // TODO: Enable the following once it not longer interferes with @(...)
        // event control checks. The introduced dummy variables make the event
        // control observe a static local variable that never changes, instead of
        // observing a module-wide signal.
        // modulePM.addPass(circt::moore::createSimplifyProceduresPass());
        modulePM.addPass(mlir::createSROA());
    }
    
    {
        // Perform a final cleanup across all modules/functions.
        auto &anyPM = pm.nestAny();
        anyPM.addPass(mlir::createMem2Reg());
        anyPM.addPass(mlir::createCSEPass());
        anyPM.addPass(mlir::createCanonicalizerPass());
    }
}

void ReadVCommand::populateMooreToCoreLowering(mlir::PassManager &pm) {
    // Perform the conversion.
    pm.addPass(circt::createConvertMooreToCorePass());

    {
        // Conversion to the core dialects likely uncovers new canonicalization
        // opportunities.
        auto &anyPM = pm.nestAny();
        anyPM.addPass(mlir::createCSEPass());
        anyPM.addPass(mlir::createCanonicalizerPass());
    }
}

void ReadVCommand::populateLLHDLowering(mlir::PassManager &pm) {
      // Inline function calls and lower SCF to CF.
    pm.addNestedPass<circt::hw::HWModuleOp>(circt::llhd::createWrapProceduralOpsPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(circt::llhd::createInlineCallsPass());
    pm.addPass(mlir::createSymbolDCEPass());

    // Simplify processes, replace signals with process results, and detect
    // registers.
    auto &modulePM = pm.nest<circt::hw::HWModuleOp>();
    // See https://github.com/llvm/circt/issues/8804.
    // modulePM.addPass(mlir::createSROA());
    modulePM.addPass(circt::llhd::createMem2RegPass());
    modulePM.addPass(circt::llhd::createHoistSignalsPass());
    modulePM.addPass(circt::llhd::createDeseqPass());
    modulePM.addPass(circt::llhd::createLowerProcessesPass());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(mlir::createCanonicalizerPass());

    // Unroll loops and remove control flow.
    modulePM.addPass(circt::llhd::createUnrollLoopsPass());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(mlir::createCanonicalizerPass());
    modulePM.addPass(circt::llhd::createRemoveControlFlowPass());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(mlir::createCanonicalizerPass());

    // Convert `arith.select` generated by some of the control flow canonicalizers
    // to `comb.mux`.
    modulePM.addPass(circt::createMapArithToCombPass());

    // Simplify module-level signals.
    modulePM.addPass(circt::llhd::createCombineDrivesPass());
    modulePM.addPass(circt::llhd::createSig2Reg());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(mlir::createCanonicalizerPass());
}

bool ReadVCommand::startWith(const std::string& str, const std::string& prefix) {
    if (str.size() < prefix.size()) {
        return false;
    }

    return str.compare(0, prefix.size(), prefix) == 0;
}

bool ReadVCommand::parseOptions(const std::vector<std::string>& args, ReadVCommandOptions& opts) {
    mlir::ModuleOp specModule = EquivFusionManager::getInstance()->getSpecModuleOp();
    mlir::ModuleOp implModule = EquivFusionManager::getInstance()->getImplModuleOp();

    for (size_t argidx = 0; argidx < args.size(); argidx++) {
        if ((args[argidx] == "--top" || args[argidx] == "-top") && argidx + 1 < args.size()) {
            opts.topModuleName = args[++argidx];
        } else if ((args[argidx] == "--spec" || args[argidx] == "-spec")) {
            opts.spec = true;
        } else if ((args[argidx] == "--impl" || args[argidx] == "-impl")) {
            opts.impl = true;
        } else if (startWith(args[argidx], "-I") || startWith(args[argidx], "--I")) {
            size_t len = startWith(args[argidx], "-I") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.includeDirs.emplace_back(args[++argidx]);
                }
            } else {
                opts.includeDirs.emplace_back(args[argidx].substr(len));
            }
        } else if ((args[argidx] == "-isystem" || args[argidx] == "--isystem") && argidx + 1 < args.size()) {
            opts.includeSystemDirs.emplace_back(args[++argidx]);
        } else if (startWith(args[argidx], "-y") || startWith(args[argidx], "--y")) {
            size_t len = startWith(args[argidx], "-y") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.libDirs.emplace_back(args[++argidx]);
                }
            } else {
                opts.libDirs.emplace_back(args[argidx].substr(len));
            }
        } else if (startWith(args[argidx], "-Y") || startWith(args[argidx], "--Y")) {
            size_t len = startWith(args[argidx], "-Y") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.libExts.emplace_back(args[++argidx]);
                }
            } else {
                opts.libExts.emplace_back(args[argidx].substr(len));
            }
        } else if ((args[argidx] == "-exclude-ext" || args[argidx] == "--exclude-ext") && argidx + 1 < args.size()) {
            opts.excludeExts.emplace_back(args[++argidx]);
        } else if (startWith(args[argidx], "-D") || startWith(args[argidx], "--D")) {
            size_t len = startWith(args[argidx], "-D") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.defines.emplace_back(args[++argidx]);
                }
            } else {
                opts.defines.emplace_back(args[argidx].substr(len));
            }
        } else if (startWith(args[argidx], "-U") || startWith(args[argidx], "--U")) {
            size_t len = startWith(args[argidx], "-U") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.undefines.emplace_back(args[++argidx]);
                }
            } else {
                opts.undefines.emplace_back(args[argidx].substr(len));
            }
        } else if ((args[argidx] == "-timescale" || args[argidx] == "--timescale") && argidx + 1 < args.size()) {
            opts.timeScale = args[++argidx];
        } else if (args[argidx] == "-allow-use-before-declare" || args[argidx] == "--allow-use-before-declare") {
            opts.allowUseBeforeDeclare = true;
        } else if (args[argidx] == "-ignore-unknown-modules" || args[argidx] == "--ignore-unknown-modules") {
            opts.ignoreUnknownModules = true;
        } else if (startWith(args[argidx], "-G") || startWith(args[argidx], "--G")) {
            size_t len = startWith(args[argidx], "-G") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.paramOverrides.emplace_back(args[++argidx]);
                }
            } else {
                opts.paramOverrides.emplace_back(args[argidx].substr(len));
            }
        } else if (startWith(args[argidx], "-l") || startWith(args[argidx], "--l")) {
            size_t len = startWith(args[argidx], "-l") ? 2 : 3;
            if (args[argidx].size() == len) {
                if (argidx + 1 < args.size()) {
                    opts.libraryFiles.emplace_back(args[++argidx]);
                }
            } else {
                opts.libraryFiles.emplace_back(args[argidx].substr(len));
            }
        } else {
            opts.inputFiles.emplace_back(args[argidx]);
        }
    }

    if (opts.inputFiles.empty()) {
        log("Command 'read_v' Failed:\n");
        log("   InputFiles is required!\n");
        return false;
    }

    if (opts.spec && opts.impl) {
        log("Command 'read_v' Failed:\n");
        log("   --spec and --impl cannot be specified at the same time!\n");
    } else if (!opts.spec && !opts.impl) {
        log("Command 'read_v' Failed:\n");
        log("   --spec or --impl must be specified!\n");
        return false;
    } else if (opts.spec) {
        if (specModule) {
            log("Command 'read_v' Failed:\n");
            log("   Specification design already specified!\n");
            return false;
        }
    } else if (opts.impl) {
        if (implModule) {
            log("Command 'read_v' Failed:\n");
            log("   Implementation design already specified!\n");
            return false;
        }
    }

    return true;
}

llvm::LogicalResult ReadVCommand::executeRTLFlow(const ReadVCommandOptions& opts) {
    circt::ImportVerilogOptions options;

    options.includeDirs = opts.includeDirs;
    options.includeSystemDirs = opts.includeSystemDirs;
    options.libDirs = opts.libDirs;
    options.libExts = opts.libExts;
    options.excludeExts = opts.excludeExts;
    options.defines = opts.defines;
    options.undefines = opts.undefines;
    if (!opts.timeScale.empty()) {
        options.timeScale = opts.timeScale;
    }
    options.allowUseBeforeDeclare = opts.allowUseBeforeDeclare;
    options.ignoreUnknownModules = opts.ignoreUnknownModules;
    if (!opts.topModuleName.empty()) {
        options.topModules.emplace_back(opts.topModuleName);
    }
    options.paramOverrides = opts.paramOverrides;
    options.libraryFiles = opts.libraryFiles;

    llvm::SourceMgr sourceMgr;
    llvm::DenseSet<llvm::StringRef> inputFilenames;
    
    for (const auto &inputFilename : opts.inputFiles) {
        if (!inputFilenames.insert(inputFilename).second) {
            logWarning("redundant input file `%s`\n", inputFilename.c_str());
            continue;
        }

        std::string errorMessage;
        auto buffer = mlir::openInputFile(inputFilename, &errorMessage);
        if (!buffer) {
            llvm::errs() << errorMessage << "\n";
            return llvm::failure();
        }

        sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    }

    mlir::MLIRContext *context = EquivFusionManager::getInstance()->getGlobalContext();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
    mlir::DefaultTimingManager tm;
    auto ts = tm.getRootScope();

    if (llvm::failed(circt::importVerilog(sourceMgr, context, ts, module.get(), &options))) {
        return llvm::failure();
    }

    mlir::PassManager pm(context);
    pm.enableVerifier(true);
    pm.enableTiming(ts);

    populateMooreTransforms(pm);
    populateMooreToCoreLowering(pm);
    populateLLHDLowering(pm);

    pm.addPass(circt::hw::createFlattenModules());
    pm.addPass(circt::comb::createEquivFusionConvertCaseToLogicalComparsionPass());

    if (llvm::failed(pm.run(module.get()))) {
        return llvm::failure();
    }
    
    if (opts.spec) {
        EquivFusionManager::getInstance()->setSpecModuleOp(module);
    } else {
        EquivFusionManager::getInstance()->setImplModuleOp(module);
    }

    return llvm::success();
}






XUANSONG_NAMESPACE_HEADER_END
