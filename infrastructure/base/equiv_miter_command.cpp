//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "infrastructure/base/command.h"
#include "libs/Tools/EquivMiter/equiv_miter.h"

XUANSONG_NAMESPACE_HEADER_START

struct EquivMiterCommand : public Command {
public:
    EquivMiterCommand() : Command("equiv_miter", "construct miter for logical equivalence check") {}
    ~EquivMiterCommand() = default;

    void execute(const std::vector<std::string>& args) override {
        EquivMiterTool::execute(args);
    }

    void help() override {
        EquivMiterTool::help(getName(), getDescription());
    }
} equivMiterCmd;

XUANSONG_NAMESPACE_HEADER_END


