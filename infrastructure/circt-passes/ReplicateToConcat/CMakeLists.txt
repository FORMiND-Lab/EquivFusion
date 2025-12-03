add_circt_library(EquivFusionPassReplicateToConcat
    replicate_to_concat.cpp

    DEPENDS
    EquivFusionReplicateToConcatIncGen

    LINK_LIBS PUBLIC
    CIRCTComb
)