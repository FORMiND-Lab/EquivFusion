add_circt_library(EquivFusionMiter
    miter.cpp

    DEPENDS
    EquivfusionMiterIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTVerif
)
