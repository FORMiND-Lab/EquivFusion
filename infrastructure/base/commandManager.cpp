#include "infrastructure/base/commandManager.h"
#include "infrastructure/log/log.h"

#include "mlir/IR/DialectRegistry.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // registerInlinerExtension     MLIRFuncInlinerExtension

XUANSONG_NAMESPACE_HEADER_START

CommandManager *CommandManager::instance_ = nullptr;

CommandManager *CommandManager::getInstance() {
    if (instance_ == nullptr) {
        instance_ = new CommandManager();
    }
    return instance_;
}

void CommandManager::registerCommand() {
    Command *cmd = firstCommand;
    while (cmd != nullptr) {
        if (hasCommand(cmd->getName())) { 
            logError("Unable to register command %s, it already exists", cmd->getName().c_str());
        }

        registeredCommands_[cmd->getName()] = cmd;
        cmd = cmd->getNextCommand();
    }
}

bool CommandManager::hasCommand(const std::string &name) const {
    return registeredCommands_.find(name) != registeredCommands_.end();
}

Command *CommandManager::getCommand(const std::string &name) const {
    auto it = registeredCommands_.find(name);
    if (it == registeredCommands_.end()) {
        return nullptr;
    }
    return it->second;
}

std::map<std::string, Command *> CommandManager::getRegisteredCommands() const {
    return registeredCommands_;
}

void CommandManager::executeCommand(const std::string &name, const std::vector<std::string> &args) { 
    if (!hasCommand(name)) { 
        log("Error: Command '%s' not found\n", name.c_str());
        return;
    }

    Command *cmd = getCommand(name);
    cmd->execute(args);
}

void CommandManager::setModuleOp(mlir::OwningOpRef<mlir::ModuleOp> &module) {
    moduleOp_ = std::move(module);
}
mlir::ModuleOp CommandManager::getModuleOp() {
    return moduleOp_ ? moduleOp_.get() : nullptr;
}

mlir::MLIRContext* CommandManager::getGlobalContext() {
    if (!globalContext_) {
        mlir::DialectRegistry registry;
        registry.insert<circt::comb::CombDialect>();
        registry.insert<circt::emit::EmitDialect>();
        registry.insert<circt::hw::HWDialect>();
        registry.insert<circt::om::OMDialect>();
        registry.insert<mlir::smt::SMTDialect>();
        registry.insert<circt::verif::VerifDialect>();
        registry.insert<mlir::func::FuncDialect>();

        mlir::func::registerInlinerExtension(registry);
        globalContext_ = std::make_unique<mlir::MLIRContext>(registry);
    }
    return globalContext_.get();
}

XUANSONG_NAMESPACE_HEADER_END
