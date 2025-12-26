//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/base/command.h"
#include "infrastructure/utils/log-util/log_util.h"
#include "infrastructure/utils/namespace_macro.h"
#include "infrastructure/managers/equivfusion_manager/equivfusionManager.h"


XUANSONG_NAMESPACE_HEADER_START

namespace {

struct SetPortCommandOptions {
    std::vector<std::string> inputPorts;
    std::vector<std::string> outputPorts;
};

} // namespace

struct SetPortCommand : public Command {
public:
    SetPortCommand() : Command("set_port", "set input or output ports.") {}
    ~SetPortCommand() = default;

    SetPortCommand(const SetPortCommand &) = delete;
    SetPortCommand &operator=(const SetPortCommand &) = delete;
    SetPortCommand(SetPortCommand &&) = delete;
    SetPortCommand &operator=(SetPortCommand &&) = delete;

    bool parseOptions(const std::vector<std::string>& args, SetPortCommandOptions& opts);

    void preExecute() override {

    }

    void execute(const std::vector<std::string>& args) override {
        SetPortCommandOptions opts;
        if (!parseOptions(args, opts)) {
            return;
        }

        EquivFusionManager::getInstance()->addInputPorts(opts.inputPorts);
        EquivFusionManager::getInstance()->addOutputPorts(opts.outputPorts);
    }

    void postExecute() override {
    }

    void help() override {
        log("\n");
        log("   OVERVIEW: %s - %s\n", getName().c_str(), getDescription().c_str());
        log("   USAGE:    set_port  <--input <portName> | --output <portName> ...>\n");
        log("   OPTIONS:\n");
        log("       --input <portName> ---------------- Specify input port name\n");
        log("       --output <portName> --------------- Specify output port name\n");
        log("\n");
    }
} setPortCommand;


bool SetPortCommand::parseOptions(const std::vector<std::string>& args, SetPortCommandOptions& opts) {
    for (size_t argidx = 0; argidx < args.size(); argidx++) {
        if ((args[argidx] == "--input" || args[argidx] == "-input") && argidx + 1 < args.size()) {
            opts.inputPorts.push_back(args[++argidx]);
        } else if ((args[argidx] == "--output" || args[argidx] == "-output") && argidx + 1 < args.size()) {
            opts.outputPorts.push_back(args[++argidx]);
        }
    }

    if (opts.inputPorts.empty() && opts.outputPorts.empty()) {
        log("Command 'set_port' Failed:\n");
        log("   At least one input or output port must be specified!\n");
        return false;
    }

    return true;
}


XUANSONG_NAMESPACE_HEADER_END
