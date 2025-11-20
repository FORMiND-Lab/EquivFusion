add_circt_library(EquivFusionPassTemporalUnroll
    temporal_unroll.cpp

    DEPENDS
    EquivFusionTemporalUnrollIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTSeq
)
