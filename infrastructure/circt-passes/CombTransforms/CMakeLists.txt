add_circt_library(EquivFusionPassCombTransforms
    decompose_concat.cpp
    replicate_to_concat.cpp

    DEPENDS
    EquivFusionPassCombTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTComb
)
