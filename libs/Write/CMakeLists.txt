aux_source_directory(base   SOURCES_BASE)
aux_source_directory(aiger  SOURCES_AIGER)
aux_source_directory(btor2  SOURCES_BTOR2)
aux_source_directory(mlir   SOURCES_MLIR)
aux_source_directory(smt    SOURCES_SMT)

add_library(EquivFusionWrite
    ${SOURCES_BASE}
    ${SOURCES_AIGER}
    ${SOURCES_BTOR2}
    ${SOURCES_MLIR}
    ${SOURCES_SMT}
)    

target_link_libraries(EquivFusionWrite
    PRIVATE
  
    MLIRExportSMTLIB
    CIRCTExportAIGER 
    CIRCTHWToBTOR2

    utils
)

