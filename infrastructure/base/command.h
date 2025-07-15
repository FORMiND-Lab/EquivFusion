#pragma once

#include <string>
#include <vector>
#include "infrastructure/utils/namespace_macro.h"

XUANSONG_NAMESPACE_HEADER_START
    
struct Command {
private: 
    std::string name_;
    std::string description_;
    Command *nextCommand_;

public:
    Command(const std::string &name, const std::string &description);
    virtual ~Command() = default;

    Command(const Command &) = delete;
    Command &operator=(const Command &) = delete;

    virtual void preExecute( ) = 0;
    virtual void execute(const std::vector<std::string> &args) = 0;
    virtual void postExecute( ) = 0;
    virtual void help() = 0;
    
    std::string getName() const;
    std::string getDescription() const;
    Command *getNextCommand() const;
};

extern Command *firstCommand;


XUANSONG_NAMESPACE_HEADER_END

