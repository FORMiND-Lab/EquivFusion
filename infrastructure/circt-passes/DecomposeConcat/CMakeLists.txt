add_circt_library(EquivFusionPassDecomposeConcat
    decompose_concat.cpp

    DEPENDS
    EquivFusionDecomposeConcatIncGen

    LINK_LIBS PUBLIC
    CIRCTComb
)
