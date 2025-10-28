add_library(EquivFusionToolEquivMiter equiv_miter.cpp)

target_link_libraries(EquivFusionToolEquivMiter
    PRIVATE

    MLIRParser
    CIRCTOMTransforms
    CIRCTEmitTransforms
    CIRCTHWTransforms
    EquivFusionPassEquivMiter

    log
)

llvm_update_compile_flags(EquivFusionToolEquivMiter)
mlir_check_all_link_libraries(EquivFusionToolEquivMiter)
