#include "infrastructure/utils/log/log.h"
#include "infrastructure/utils/path/path.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"
#include "infrastructure/base/command.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

XUANSONG_NAMESPACE_HEADER_START

struct WriteMLIROptions {
    std::string outputFilename {"-"};
    enum class ModuleEnum {
        IMPL,
        SPEC,
        MITER
    };
    ModuleEnum module {ModuleEnum::IMPL};
};

struct WriteMLIRCommand : public Command {
public:
    WriteMLIRCommand() : Command("write_mlir", "write mlir to file") {}
    ~WriteMLIRCommand() = default;

    WriteMLIRCommand(const WriteMLIRCommand &) = delete;
    WriteMLIRCommand &operator=(const WriteMLIRCommand &) = delete;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        WriteMLIROptions opts;
        if (!parseOptions(args, opts)) {
            log("[write_mlir]: parse options failed\n\n");
            return;
        }

        if (!run(opts)) {
            log("[write_mlir]: run failed\n\n");
            return;
        }
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    %s [--module spec | impl | miter] <outputfile>\n", getName().c_str());
        log("   OPTIONS:\n");
        log("       --module ------------------------------------ Write module [spec, impl, miter]. \n");
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
        if ((arg == "--module" || arg == "-module") && idx + 1 < args.size()) {
            auto val = args[++idx];
            if (val == "spec") {
                opts.module = WriteMLIROptions::ModuleEnum::SPEC;
            } else if (val == "impl") {
                opts.module = WriteMLIROptions::ModuleEnum::IMPL;
            } else if (val == "miter") {
                opts.module = WriteMLIROptions::ModuleEnum::MITER;
            } else {
                log("Wrong options val of -mode.\n");
                return false;
            }
        } else {
            opts.outputFilename = arg;
            Utils::PathUtil::expandTilde(opts.outputFilename);
        }
    }
    return true;
}

/// Run write_mlir with WriteMLIROptions
bool WriteMLIRCommand::run(const WriteMLIROptions& opts) {
    // Check module
    mlir::ModuleOp module;
    switch (opts.module) {
    case WriteMLIROptions::ModuleEnum::SPEC:
        module = EquivFusionManager::getInstance()->getSpecModuleOp();
        break;
    case WriteMLIROptions::ModuleEnum::IMPL:
        module = EquivFusionManager::getInstance()->getImplModuleOp();
        break;
    case WriteMLIROptions::ModuleEnum::MITER:
        module = EquivFusionManager::getInstance()->getMergedModuleOp();
        break;
    default:
        assert(0 && "Wrong mode.\n");
    }
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
