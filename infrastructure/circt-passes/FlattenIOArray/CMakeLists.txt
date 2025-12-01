add_circt_library(EquivFusionFlattenIOArray
    flatten_io_array.cpp

    DEPENDS
    EquivFusionFlattenIOArrayIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
)
