add_circt_library(EquivFusionDecomposeConcat
    decompose_concat.cpp

    DEPENDS
    EquivFusionDecomposeConcatIncGen

    LINK_LIBS PUBLIC
    CIRCTComb
)
