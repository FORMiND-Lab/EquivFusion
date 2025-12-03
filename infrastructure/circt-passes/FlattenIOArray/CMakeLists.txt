add_circt_library(EquivFusionPassFlattenIOArray
    flatten_io_array.cpp

    DEPENDS
    EquivFusionFlattenIOArrayIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
)
