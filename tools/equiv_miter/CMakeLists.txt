add_llvm_executable(equiv_miter equiv_miter.cpp)

target_link_libraries(equiv_miter
  PRIVATE
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

  EquivFusionMiter 
  CIRCTHWTransforms
  CIRCTCombToAIG
  CIRCTAIGTransforms
  CIRCTExportAIGER
  CIRCTHWToBTOR2
)

llvm_update_compile_flags(equiv_miter)
mlir_check_all_link_libraries(equiv_miter)


