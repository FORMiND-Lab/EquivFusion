add_circt_library(EquivFusionMiter
    miter.cpp

    DEPENDS
    EquivFusionMiterIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTVerif
)
