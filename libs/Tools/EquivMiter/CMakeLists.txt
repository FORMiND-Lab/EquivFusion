add_library(EquivFusionToolEquivMiter equiv_miter.cpp)

target_link_libraries(EquivFusionToolEquivMiter
    PRIVATE

    CIRCTOMTransforms
    CIRCTEmitTransforms
    CIRCTHWTransforms
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
    
    EquivFusionPassDecomposeConcat
    EquivFusionManager
    CIRCTArcTransforms
    CIRCTHWToBTOR2    

    utils
)

llvm_update_compile_flags(EquivFusionToolEquivMiter)
mlir_check_all_link_libraries(EquivFusionToolEquivMiter)
