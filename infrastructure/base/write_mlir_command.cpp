//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/utils/path-util/path_util.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/base/command.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

struct WriteMLIROptions {
    std::string outputFilename {"-"};
    ModuleTypeEnum moduleType {ModuleTypeEnum::UNKNOWN};
};

struct WriteMLIRCommand : public Command {
public:
    WriteMLIRCommand() : Command("write_mlir", "write mlir to file") {}
    ~WriteMLIRCommand() = default;

    WriteMLIRCommand(const WriteMLIRCommand &) = delete;
    WriteMLIRCommand &operator=(const WriteMLIRCommand &) = delete;

    void execute(const std::vector<std::string>& args) override {
        WriteMLIROptions opts;
        if (!parseOptions(args, opts)) {
            return;
        }

        if (!run(opts)) {
            log("[write_mlir]: run failed\n\n");
            return;
        }
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    %s [--spec | --impl | --miter] <outputfile>\n", getName().c_str());
        log("   OPTIONS:\n");
        log("       --spec ------------------------------------ specification module\n");
        log("       --impl ------------------------------------ implementation module \n");
        log("       --miter ----------------------------------- miter module\n");
        log("   Example:  %s test.output", getName().c_str());
        log("\n\n");
    }

private:
    bool parseOptions(const std::vector<std::string>& args, WriteMLIROptions& opts);
    bool run(const WriteMLIROptions& opts);

} writeMLIRCmd;


/// Parse options to WriteMLIROptions
bool WriteMLIRCommand::parseOptions(const std::vector<std::string>& args, WriteMLIROptions& opts) {
    for (size_t idx = 0; idx < args.size(); idx++) {
        auto arg = args[idx];
        if (arg == "--spec" || arg == "-spec") {
            opts.moduleType = ModuleTypeEnum::SPEC;
        } else if (arg == "--impl" || arg == "-impl") {
            opts.moduleType = ModuleTypeEnum::IMPL;
        } else if (arg == "--miter" || arg == "-miter") {
            opts.moduleType = ModuleTypeEnum::MITER;
        } else {
            opts.outputFilename = arg;
            Utils::PathUtil::expandTilde(opts.outputFilename);
        }
    }

    if (opts.moduleType == ModuleTypeEnum::UNKNOWN) {
        log("[write_mlir]: please specify module type.\n");
        return false;
    }
    return true;
}

/// Run write_mlir with WriteMLIROptions
bool WriteMLIRCommand::run(const WriteMLIROptions& opts) {
    // Check module
    mlir::ModuleOp module = EquivFusionManager::getInstance()->getModuleOp(opts.moduleType);
    if (!module) {
        return true;
    }

    // Open outputfile
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
    std::string errorMessage;
    outputFile.emplace(mlir::openOutputFile(opts.outputFilename, &errorMessage));
    if (!outputFile.value()) {
        log("[write_mlir]: open output file failed[%s]\n\n", errorMessage.c_str());
        return false;
    }

    // Print mlir to output file
    module.print(outputFile.value()->os());
    outputFile.value()->keep();

     return true;
}

XUANSONG_NAMESPACE_HEADER_END
