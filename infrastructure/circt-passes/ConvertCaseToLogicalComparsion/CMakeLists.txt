add_circt_library(EquivFusionConvertCaseToLogicalComparison
    convert_case_to_logical_comparsion.cpp

    DEPENDS
    EquivFusionConvertCaseToLogicalComparisonIncGen

    LINK_LIBS PUBLIC
    CIRCTComb
    MLIRTransformUtils
    MLIRIR
)