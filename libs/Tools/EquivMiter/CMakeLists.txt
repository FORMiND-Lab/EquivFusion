add_library(EquivFusionToolEquivMiter equiv_miter.cpp)

target_link_libraries(EquivFusionToolEquivMiter
    PRIVATE

    CIRCTOMTransforms
    CIRCTEmitTransforms
    CIRCTHWTransforms
    EquivFusionPassFlattenIOArray
    EquivFusionPassEquivMiter
    CIRCTHWToSMT
    CIRCTCombToSMT
    CIRCTVerifToSMT
    CIRCTSupport
    MLIRExportSMTLIB
    MLIRIR

    CIRCTCombToSynth
    CIRCTSynthTransforms
    CIRCTExportAIGER

    EquivFusionPassReplicateToConcat
    EquivFusionPassDecomposeConcat
    CIRCTArcTransforms
    CIRCTHWToBTOR2

    EquivFusionManager
    EquivFusionUtils
)

llvm_update_compile_flags(EquivFusionToolEquivMiter)
mlir_check_all_link_libraries(EquivFusionToolEquivMiter)
