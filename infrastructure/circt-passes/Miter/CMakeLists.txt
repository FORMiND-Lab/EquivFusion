add_circt_library(EquivFusionPassEquivMiter
    miter.cpp

    DEPENDS
    EquivFusionMiterIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTVerif
)
