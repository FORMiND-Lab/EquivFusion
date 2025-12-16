add_circt_library(EquivFusionFuncToHWModule
    func_to_hw_module.cpp

    DEPENDS
    EquivFusionFuncToHWModuleIncGen

    LINK_LIBS PUBLIC
    MLIRControlFlowDialect
    MLIRFuncDialect
    CIRCTHW
    CIRCTComb
)


