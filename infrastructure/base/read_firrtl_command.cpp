//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/base/command.h"
#include "libs/Tools/ReadFIRRTL/read_firrtl.h"

XUANSONG_NAMESPACE_HEADER_START

struct ReadFIRRTLCommand : public Command {
    ReadFIRRTLCommand() : Command("read_firrtl", "read .fir file") {}
    ~ReadFIRRTLCommand() = default;

    ReadFIRRTLCommand(const ReadFIRRTLCommand &) = delete;
    ReadFIRRTLCommand &operator=(const ReadFIRRTLCommand &) = delete;
    ReadFIRRTLCommand(ReadFIRRTLCommand &&) = delete;
    ReadFIRRTLCommand &operator=(ReadFIRRTLCommand &&) = delete;

    void preExecute() override {
    }

    void execute(const std::vector<std::string>& args) override {
        ReadFIRRTLTool::execute(args);
    }

    void postExecute() override {
    }

    void help() override {
        ReadFIRRTLTool::help(getName(), getDescription());
    }

} readFIRRTLCmd;

XUANSONG_NAMESPACE_HEADER_END // namespace XuanSong