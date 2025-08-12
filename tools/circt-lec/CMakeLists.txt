if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_compile_definitions(CIRCT_LEC_ENABLE_JIT)
  set(CIRCT_LEC_JIT_LLVM_COMPONENTS native)
  set(CIRCT_LEC_JIT_DEPS
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
  )
endif()

message(STATUS "CIRCT_LEC_JIT_LLVM_COMPONENTS: ${CIRCT_LEC_JIT_LLVM_COMPONENTS}") 

set(LLVM_LINK_COMPONENTS Support ${CIRCT_LEC_JIT_LLVM_COMPONENTS})

add_llvm_executable(EquivFusionLEC circt-lec.cpp)

target_link_libraries(EquivFusionLEC
  PRIVATE
  CIRCTComb
  CIRCTCombToSMT
  CIRCTEmitTransforms
  CIRCTHW
  CIRCTHWToSMT
  CIRCTLECTransforms
  CIRCTOMTransforms
  CIRCTSMTToZ3LLVM
  CIRCTSupport
  CIRCTVerif
  CIRCTVerifToSMT
  LLVMSupport
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRSMT
  MLIRTargetLLVMIRExport
  MLIRExportSMTLIB

  ${CIRCT_LEC_JIT_DEPS}
)

llvm_update_compile_flags(EquivFusionLEC)
mlir_check_all_link_libraries(EquivFusionLEC)
