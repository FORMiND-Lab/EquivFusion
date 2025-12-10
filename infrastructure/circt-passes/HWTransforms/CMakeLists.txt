add_circt_library(EquivFusionPassHWTransforms
    flatten_io_array.cpp
    flatten_array_slice.cpp
    flatten_struct_explode.cpp
    hw_aggregate_to_comb.cpp

    DEPENDS
    EquivFusionPassHWTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
)
