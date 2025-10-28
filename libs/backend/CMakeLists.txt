add_library(EquivFusionBackend
    backend_base.cpp
    write_btor2.cpp
    write_smt.cpp
    write_aiger.cpp
)

target_link_libraries(EquivFusionBackend
    PRIVATE
    CIRCTHW
   
    CIRCTHWTransforms
    EquivFusionPassDecomposeConcat
    CIRCTArcTransforms
    CIRCTHWToBTOR2

    CIRCTHWToSMT
    CIRCTCombToSMT
    CIRCTVerifToSMT
    CIRCTSupport
    MLIRExportSMTLIB

    CIRCTCombToAIG
    CIRCTAIGTransforms
    CIRCTExportAIGER
)

