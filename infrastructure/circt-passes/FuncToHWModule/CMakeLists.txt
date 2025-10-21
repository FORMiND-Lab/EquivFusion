add_circt_library(FuncToHWModule
    func_to_hw_module.cpp

    DEPENDS
    FuncToHWModuleIncGen

    LINK_LIBS PUBLIC
    MLIRControlFlowDialect
    MLIRFuncDialect
    CIRCTHW
    CIRCTComb
)


