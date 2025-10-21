add_library(EquivFusionToolEquivMiter equiv_miter_tool.cpp)

target_link_libraries(EquivFusionToolEquivMiter
    PUBLIC
    CIRCTComb
    CIRCTCombToSMT
    CIRCTEmitTransforms
    CIRCTHW
    CIRCTHWToSMT
    CIRCTOMTransforms
    CIRCTSupport
    CIRCTVerif
    CIRCTVerifToSMT

    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRSMT
    MLIRExportSMTLIB

    CIRCTHWTransforms
    CIRCTCombToAIG
    CIRCTAIGTransforms
    CIRCTExportAIGER
    CIRCTHWToBTOR2

    CIRCTArcTransforms

    EquivFusionMiter
    EquivFusionDecomposeConcat
)

llvm_update_compile_flags(EquivFusionToolEquivMiter)
mlir_check_all_link_libraries(EquivFusionToolEquivMiter)


