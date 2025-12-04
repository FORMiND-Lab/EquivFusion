add_circt_library(EquivFusionPassHWTransforms
    flatten_io_array.cpp

    DEPENDS
    EquivFusionPassHWTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
)
