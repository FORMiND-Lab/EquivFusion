add_circt_library(EquivFusionPassHWTransforms
    flatten_io_array.cpp
    flatten_array_slice.cpp

    DEPENDS
    EquivFusionPassHWTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
)
