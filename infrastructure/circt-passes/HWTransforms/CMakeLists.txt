add_circt_library(EquivFusionPassHWTransforms
    flatten_io_array.cpp
    hw_aggregate_ops_convert.cpp
    hw_aggregate_to_comb.cpp
    init_post_defined_operands.cpp

    DEPENDS
    EquivFusionPassHWTransformsIncGen

    LINK_LIBS PUBLIC
    CIRCTHW
)
