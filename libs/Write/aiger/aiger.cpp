//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"

#include "libs/Write/aiger/aiger.h"

#include "circt/Dialect/HW/HWOps.h"

#include "circt/Conversion/ExportAIGER.h"   // exportAIGER                      CIRCTExportAIGER

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteAIGERImpl::run(const std::vector<std::string>& args) {
    // Parse options
    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_aiger]: parser options failed\n\n");
        return false;
    }

    // Check module
    mlir::ModuleOp module = EquivFusionManager::getInstance()->getMergedModuleOp();
    if (!module) {
        return true;
    }

    // Open output file
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_aiger]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    // Print AIGER to output file
    auto ops = module.getOps<circt::hw::HWModuleOp>();
    if (ops.empty() || std::next(ops.begin()) != ops.end())
        return false;

    circt::aiger::ExportAIGEROptions exportAIGEROpts = {true, true};
    if (failed(circt::aiger::exportAIGER(*ops.begin(), outputFile.value()->os(), &exportAIGEROpts))) {
        log("[write_aiger]: run exportAIGER failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}

XUANSONG_NAMESPACE_HEADER_END
