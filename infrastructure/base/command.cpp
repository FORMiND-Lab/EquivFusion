//===-----------------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------------===//

#include "infrastructure/base/command.h"
#include <iostream>


XUANSONG_NAMESPACE_HEADER_START

Command *firstCommand = nullptr;

Command::Command(const std::string &name, const std::string &description) 
    : name_(name), description_(description) {
        nextCommand_ = firstCommand;
        firstCommand = this;
    }

std::string Command::getName() const {
    return name_;
}

std::string Command::getDescription() const {
    return description_;
}

Command *Command::getNextCommand() const {
    return nextCommand_;
}


XUANSONG_NAMESPACE_HEADER_END
