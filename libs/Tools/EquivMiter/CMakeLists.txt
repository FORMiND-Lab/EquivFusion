add_library(EquivFusionToolEquivMiter equiv_miter.cpp)

target_link_libraries(EquivFusionToolEquivMiter
    PRIVATE

    CIRCTOMTransforms
    CIRCTEmitTransforms
    CIRCTHWTransforms

    CIRCTHWToSMT
    CIRCTCombToSMT
    CIRCTVerifToSMT
    CIRCTSupport
    MLIRExportSMTLIB
    MLIRIR

    CIRCTCombToSynth
    CIRCTSynthTransforms
    CIRCTExportAIGER

    CIRCTArcTransforms
    CIRCTHWToBTOR2

    EquivFusionManager
    EquivFusionUtils

    EquivFusionPassEquivMiter
    EquivFusionPassCombTransforms
    EquivFusionPassHWTransforms
)

llvm_update_compile_flags(EquivFusionToolEquivMiter)
mlir_check_all_link_libraries(EquivFusionToolEquivMiter)
