add_llvm_executable(equivfusion-hls equivfusion-hls.cpp)

set(LLVM_LINK_COMPONENTS Support)

target_link_libraries(equivfusion-hls 
  PRIVATE

  CIRCTHW
  CIRCTHWTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTTransforms

  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRControlFlowDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRAffineToStandard

  FuncToHWModule
  EquivFusionRemoveRedundantFunc
  log
)

target_include_directories(equivfusion-hls PUBLIC ${INCLUDE_DIRS})

llvm_update_compile_flags(equivfusion-hls)
mlir_check_all_link_libraries(equivfusion-hls)
