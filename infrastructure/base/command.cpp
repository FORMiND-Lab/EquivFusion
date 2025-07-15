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
