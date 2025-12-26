//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"

#include "libs/Write/btor2/btor2.h"

#include "circt/Conversion/HWToBTOR2.h"             // createConvertHWToBTOR2Pass()          CIRCTHWToBTOR2

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

bool WriteBTOR2Impl::run(const std::vector<std::string>& args) {
    // Parse Options
    WriteImplOptions opts;
    if (!parseOptions(args, opts)) {
        log("[write_btor2]: parser options failed\n\n");
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
        log("[write_btor2]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    // Print btor2 to output file
    mlir::MLIRContext *context = EquivFusionManager::getInstance()->getGlobalContext();
    mlir::PassManager pm(context);
    pm.addPass(circt::createConvertHWToBTOR2Pass(outputFile.value()->os()));
    if (failed(pm.run(module))) {
        log("[write_btor2]: run PassManager failed\n\n");
        return false;
    }

    outputFile.value()->keep();
    return true;
}


XUANSONG_NAMESPACE_HEADER_END

