#pragma once

#include <map>
#include "infrastructure/base/command.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/MLIRContext.h"

XUANSONG_NAMESPACE_HEADER_START

struct CommandManager {
private: 
    static CommandManager *instance_;
    std::map<std::string, Command *> registeredCommands_;
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp_;
    std::unique_ptr<mlir::MLIRContext> globalContext_;

public: 
    CommandManager() = default;
    ~CommandManager() = default;

    CommandManager(const CommandManager &) = delete;
    CommandManager &operator=(const CommandManager &) = delete;

    static CommandManager *getInstance();

    void registerCommand();

    bool hasCommand(const std::string &name) const;

    Command *getCommand(const std::string &name) const;

    std::map<std::string, Command *> getRegisteredCommands() const;

    void executeCommand(const std::string &name, const std::vector<std::string> &args);

    void setModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module);
    mlir::ModuleOp getModuleOp();
    mlir::MLIRContext* getGlobalContext();
};

XUANSONG_NAMESPACE_HEADER_END
