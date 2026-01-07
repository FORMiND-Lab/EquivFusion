//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Firtool/Firtool.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "mlir/Support/FileUtilities.h"

#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/utils/path-util/path_util.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "libs/Tools/ReadFIRRTL/read_firrtl.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

XUANSONG_NAMESPACE_HEADER_START
namespace ReadFIRRTLTool {

using InfoLocHandling = firrtl::FIRParserOptions::InfoLocHandling;
enum class LayerSpecializationOpt { None, Enable, Disable };

struct ReadFIRRTLOptions {
    bool printIR = false;
    ModuleTypeEnum moduleType = ModuleTypeEnum::UNKNOWN;
    std::string topModuleName;

    std::string inputFilename;
    std::vector<std::string> includeDirs;
    InfoLocHandling infoLocHandling = InfoLocHandling::PreferInfo;
    bool scalarizePublicModules = true;
    bool scalarizeIntModules = false;
    bool scalarizeExtModules = true;
    std::string highFIRRTLPassPlugin;
    std::string lowFIRRTLPassPlugin;
    std::string hwPassPlugin;
    std::vector<std::string> inputAnnotationFilenames;
    std::vector<std::string> enableLayers;
    std::vector<std::string> disableLayers;
    LayerSpecializationOpt defaultLayerSpecialization = LayerSpecializationOpt::None;
    std::vector<std::string> selectInstanceChoice;

    circt::firrtl::PreserveAggregate::PreserveMode preserveAggregate = circt::firrtl::PreserveAggregate::None;
};

static bool parseOptions(const std::vector<std::string> &args, ReadFIRRTLOptions &options) {
    bool spec = false;
    bool impl = false;

    for (size_t argidx = 0; argidx < args.size(); argidx++) {
        std::string arg = args[argidx];
        if (arg == "--print-ir" || arg == "-print-ir") {
            options.printIR = true;
        } else if (arg == "--top" || arg == "-top") {
            options.topModuleName = args[++argidx];
        } else if (arg == "--spec" || arg == "-spec") {
            spec = true;
        } else if (arg == "--impl" || arg == "-impl") {
            impl = true;
        } else if (arg == "--include-dir" || arg == "-include-dir") {
            std::string val = args[++argidx];
            Utils::PathUtil::expandTilde(val);
            options.includeDirs.emplace_back(val);
        } else if (arg == "--ignore-info-locators" || arg =="-ignore-info-locators" ) {
            options.infoLocHandling = InfoLocHandling::IgnoreInfo;
        } else if (arg == "--fuse-info-locators" || arg =="-fuse-info-locators" ) {
            options.infoLocHandling = InfoLocHandling::FusedInfo;
        } else if (arg == "--prefer-info-locators" || arg =="-iprefer-info-locators" ) {
            options.infoLocHandling = InfoLocHandling::PreferInfo;
        } else if (arg == "--scalarize-public-modules" || arg == "-scalarize-public-modules") {
            std::string val = args[++argidx];
            if (val == "true") options.scalarizePublicModules = true;
            else if (val == "false") options.scalarizePublicModules = false;
        } else if (arg == "--scalarize-internal-modules" || arg == "-scalarize-internal-modules") {
            std::string val = args[++argidx];
            if (val == "true") options.scalarizeIntModules = true;
            else if (val == "false") options.scalarizeIntModules = false;
        } else if (arg == "--scalarize-ext-modules" || arg == "-scalarize-ext-modules") {
            std::string val = args[++argidx];
            if (val == "true") options.scalarizeExtModules = true;
            else if (val == "false") options.scalarizeExtModules = false;
        } else if (arg == "--high-firrtl-pass-plugin" || arg == "-high-firrtl-pass-plugin") {
            options.highFIRRTLPassPlugin = args[++argidx];
        } else if (arg == "--low-firrtl-pass-plugin" || arg == "-low-firrtl-pass-plugin") {
            options.lowFIRRTLPassPlugin = args[++argidx];
        } else if (arg == "--hw-pass-plugin" || arg == "-hw-pass-plugin") {
            options.hwPassPlugin = args[++argidx];
        } else if (arg == "--annotation-file" || arg == "-annotation-file") {
            std::string val = args[++argidx];
            Utils::PathUtil::expandTilde(val);
            options.inputAnnotationFilenames.emplace_back(val);
        } else if (arg == "--enable-layers" || arg == "-enable-layers") {
            std::string val = args[++argidx];
            options.enableLayers.emplace_back(val);
        } else if (arg == "--disable-layers" || arg == "-disable-layers") {
            std::string val = args[++argidx];
            options.disableLayers.emplace_back(val);
        } else if (arg == "--default-layer-specialization" || arg == "-default-layer-specialization") {
            std::string val = args[++argidx];
            if (val == "disable") {
                options.defaultLayerSpecialization = LayerSpecializationOpt::Disable;
            } else if (val == "enable") {
                options.defaultLayerSpecialization = LayerSpecializationOpt::Enable;
            }
        } else if (arg == "--select-instance-choice" || arg == "-select-instance-choice") {
            std::string val = args[++argidx];
            options.disableLayers.emplace_back(val);
        } else if (arg == "--preserve-aggregate" || arg == "-preserve-aggregate") {
            std::string val = args[++argidx];
            if (val == "1d-vec") {
                options.preserveAggregate = circt::firrtl::PreserveAggregate::OneDimVec;
            } else if (val == "vec") {
                options.preserveAggregate = circt::firrtl::PreserveAggregate::Vec;
            } else if (val == "all") {
                options.preserveAggregate = circt::firrtl::PreserveAggregate::All;
            } else if (val == "none") {
                options.preserveAggregate = circt::firrtl::PreserveAggregate::None;
            }
        } else {
            options.inputFilename = arg;
            Utils::PathUtil::expandTilde(options.inputFilename);
        }
    }

    if (spec == impl) {
        log("");
        return false;
    }
    options.moduleType = spec ? ModuleTypeEnum::SPEC : ModuleTypeEnum::IMPL;
    return true;
}

static LogicalResult exeuteFirtool(MLIRContext &context, ReadFIRRTLOptions &options) {
    // Create the timing manager we use to sample execution times.
    DefaultTimingManager tm;
    auto ts = tm.getRootScope();

    // Set up the input file.
    std::string errorMessage;
    auto input = openInputFile(options.inputFilename, &errorMessage);
    if (!input) {
        llvm::errs() << errorMessage << "\n";
        return failure();
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
    sourceMgr.setIncludeDirs(options.includeDirs);

    // Add the annotation file if one was explicitly specified.
    unsigned numAnnotationFiles = 0;
    for (const auto &inputAnnotationFilename : options.inputAnnotationFilenames) {
        std::string annotationFilenameDetermined;
        if (!sourceMgr.AddIncludeFile(inputAnnotationFilename, llvm::SMLoc(),
                                      annotationFilenameDetermined)) {
            llvm::errs() << "cannot open input annotation file '"
                         << inputAnnotationFilename
                         << "': No such file or directory\n";
            return failure();
        }
        ++numAnnotationFiles;
    }

    // Parse the input.
    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto parserTimer = ts.nest("FIR Parser");
    firrtl::FIRParserOptions firParserOptions;
    firParserOptions.infoLocatorHandling = options.infoLocHandling;
    firParserOptions.numAnnotationFiles = numAnnotationFiles;
    firParserOptions.scalarizePublicModules = options.scalarizePublicModules;
    firParserOptions.scalarizeInternalModules = options.scalarizeIntModules;
    firParserOptions.scalarizeExtModules = options.scalarizeExtModules;
    firParserOptions.enableLayers = options.enableLayers;
    firParserOptions.disableLayers = options.disableLayers;
    firParserOptions.selectInstanceChoice = options.selectInstanceChoice;

    switch (options.defaultLayerSpecialization) {
        case LayerSpecializationOpt::None:
            firParserOptions.defaultLayerSpecialization = std::nullopt;
            break;
        case LayerSpecializationOpt::Enable:
            firParserOptions.defaultLayerSpecialization = firrtl::LayerSpecialization::Enable;
            break;
        case LayerSpecializationOpt::Disable:
            firParserOptions.defaultLayerSpecialization = firrtl::LayerSpecialization::Disable;
            break;
    }

    module = importFIRFile(sourceMgr, &context, parserTimer, firParserOptions);
    if (!module)
        return failure();

    PassManager pm(&context);
    pm.enableVerifier(true);
    pm.enableTiming(ts);
    EquivFusionManager::getInstance()->configureIRPrinting(pm, options.printIR);

    // TODO(taomengxia: 2025/12/26): use default firtoolOptions,
    firtool::FirtoolOptions firtoolOptions;
    firtoolOptions.setPreserveAggregate(options.preserveAggregate);

    if (failed(firtool::populatePreprocessTransforms(pm, firtoolOptions)))
        return failure();

    if (!options.highFIRRTLPassPlugin.empty())
        if (failed(parsePassPipeline(StringRef(options.highFIRRTLPassPlugin), pm)))
            return failure();

    if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, firtoolOptions)))
        return failure();

    if (!options.lowFIRRTLPassPlugin.empty())
        if (failed(parsePassPipeline(StringRef(options.lowFIRRTLPassPlugin), pm)))
            return failure();

    if (failed(firtool::populateLowFIRRTLToHW(pm, firtoolOptions, options.inputFilename)))
        return failure();

    if (!options.hwPassPlugin.empty())
        if (failed(parsePassPipeline(StringRef(options.hwPassPlugin), pm)))
            return failure();

    if (failed(pm.run(module.get())))
        return failure();

    EquivFusionManager::getInstance()->setModuleOp(module, options.moduleType);
    return success();
}

void help(const std::string &name, const std::string &description) {
    log("\n");
    log("   OVERVIEW: %s - %s\n", name.c_str(), description.c_str());
    log("   USAGE:    %s [options] <--spec | --impl> <inputFiles>\n", name.c_str());
    log("   OPTIONS:\n");
    log("       --print-ir --------------------------------- Print IR after pass\n");
    log("       --top <topModuleName> ---------------------- Specify top module name\n");
    log("       --spec ------------------------------------- Design is specification\n");
    log("       --impl ------------------------------------- Design is implementation\n");
    log("       --include-dir ------------------------------ Directory to search in when resolving source references\n");
    log("       Location tracking:\n");
    log("               --ignore-info-locators ------------- Ignore the @info locations in the .fir file\n");
    log("               --fuse-info-locators --------------- @info locations are fused with .fir locations\n");
    log("               --prefer-info-locators ------------- Use @info locations when present, fallback to .fir locations\n");
    log("       --scalarize-public-modules ----------------- Scalarize all public modules\n");
    log("       --scalarize-internal-modules --------------- Scalarize the ports of any internal modules\n");
    log("       --scalarize-ext-modules -------------------- Scalarize the ports of any external modules\n");
    log("       --high-firrtl-pass-plugin ------------------ Insert passes after parsing FIRRTL. Specify passes with MLIR textual format.\n");
    log("       --low-firrtl-pass-plugin ------------------- Insert passes before lowering to HW. Specify passes with MLIR textual format.\n");
    log("       --hw-pass-plugin --------------------------- Insert passes after lowering to HW. Specify passes with MLIR textual format.\n");
    log("       --annotation-file -------------------------- Optional input annotation file\n");
    log("       --enable-layers ---------------------------- enable these layers permanently\n");
    log("       --disable-layers --------------------------- disable these layers permanently\n");
    log("       --default-layer-specialization ------------- The default specialization for layers\n");
    log("               none ------------------------------- Layers are unchanged\n");
    log("               disable ---------------------------- Layers are disabled\n");
    log("               enable ----------------------------- Layers are enabled\n");
    log("       --select-instance-choice ------------------- Options to specialize instance choice, in option=case format\n");
    log("       --preserve-aggregate ----------------------- Specify input file format\n");
    log("\n");
}

bool execute(const std::vector<std::string> &args) {
    ReadFIRRTLOptions options;
    if (!parseOptions(args, options)) {
        log ("[read_firrtl]: parse options failed!\n\n");
        return false;
    }

    mlir::MLIRContext *context = EquivFusionManager::getInstance()->getGlobalContext();
    if (failed(exeuteFirtool(*context, options))) {
        log ("[read_firrtl]: execute failed!\n\n");
        return false;
    }

    return true;
}

} // namespace ReadFIRRTLTool
XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong
