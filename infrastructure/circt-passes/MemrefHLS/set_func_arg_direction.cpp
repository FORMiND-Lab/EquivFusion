//===----------------------------------------------------------------------===//
//
// Part of the EquivFusion Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"

#include "llvm/Support/Casting.h"
#include "llvm/ADT/SmallVector.h"

#include "circt-passes/MemrefHLS/Passes.h"

namespace circt {

#define GEN_PASS_DEF_EQUIVFUSIONSETFUNCARGDIRECTIONPASS
#include "circt-passes/MemrefHLS/Passes.h.inc"

}

namespace {

struct EquivFusionSetFuncArgDirectionPass : public circt::impl::EquivFusionSetFuncArgDirectionPassBase<EquivFusionSetFuncArgDirectionPass> {
    using circt::impl::EquivFusionSetFuncArgDirectionPassBase<EquivFusionSetFuncArgDirectionPass>::EquivFusionSetFuncArgDirectionPassBase;
    
    void runOnOperation() override;
};

}

void EquivFusionSetFuncArgDirectionPass::runOnOperation() {
    mlir::func::FuncOp funcOp = llvm::dyn_cast<mlir::func::FuncOp>(getOperation());
    std::set<std::string> inputPortsSet(this->inputPorts.begin(), inputPorts.end());
    std::set<std::string> outputPortsSet(this->outputPorts.begin(), outputPorts.end());

    std::optional<mlir::ArrayAttr> existArgAttrs = funcOp.getArgAttrs();
    llvm::SmallVector<mlir::Attribute> newAttrs;

    mlir::OpBuilder builder(&getContext());

    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        mlir::NamedAttrList newArgAttr;
        if (existArgAttrs.has_value() && i < (*existArgAttrs).size()) {
            mlir::DictionaryAttr argAttr = llvm::dyn_cast<mlir::DictionaryAttr>((*existArgAttrs)[i]);
            if (argAttr) {
                newArgAttr = mlir::NamedAttrList(argAttr.getValue().begin(), argAttr.getValue().end());
            }
        }

        std::string portName;
        if (mlir::StringAttr argName = llvm::dyn_cast_or_null<mlir::StringAttr>(newArgAttr.get("polygeist.param_name"))) {
            portName = argName.getValue().str();
        }
        
        if (outputPortsSet.find(portName) != outputPortsSet.end()) {
            newArgAttr.set("equivfusion.direction", builder.getStringAttr("out"));
        } else {
            newArgAttr.set("equivfusion.direction", builder.getStringAttr("in"));
        }

        newAttrs.push_back(builder.getDictionaryAttr(newArgAttr));
    }

    funcOp.setArgAttrsAttr(builder.getArrayAttr(newAttrs));
}


