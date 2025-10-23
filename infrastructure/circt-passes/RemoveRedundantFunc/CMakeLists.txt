add_circt_library(EquivFusionRemoveRedundantFunc
    equivfusion_remove_redundant_func.cpp

    DEPENDS
    EquivFusionRemoveRedundantFuncIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
)
