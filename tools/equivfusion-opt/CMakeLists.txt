set(LLVM_LINK_COMPONENTS  Support)

add_llvm_executable(equivfusion-opt equivfusion-opt.cpp)

get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)

target_link_libraries(equivfusion-opt
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}

    MLIRIR
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIROptLib
    MLIRParser
    MLIRFuncDialect
    MLIRSupport
    MLIRTransforms
    MLIRSCFDialect
    MLIREmitCDialect
    MLIRFuncInlinerExtension
    MLIRVectorDialect
    MLIRIndexDialect
    MLIRLLVMIRTransforms

    EquivFusionPassCombTransforms
    EquivFusionConvertCaseToLogicalComparison
    EquivFusionFuncToHWModule
    EquivFusionPassHWTransforms
    EquivFusionMemrefHLS
    EquivFusionPassEquivMiter
    EquivFusionRemoveRedundantFunc
    EquivFusionPassTemporalUnroll
)

llvm_update_compile_flags(equivfusion-opt)
mlir_check_all_link_libraries(equivfusion-opt)