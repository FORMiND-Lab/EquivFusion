aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(base STATIC ${SRC_FILES})

target_include_directories(base PRIVATE ${INCLUDE_DIRS})
target_include_directories(base PRIVATE ${READLINE_INCLUDE_DIRECTORY})

target_link_libraries(base
    PUBLIC
    ${READLINE_LIBRARY}

    MLIRFuncInlinerExtension
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRAffineDialect
    MLIRSupport
    MLIRParser
    MLIRIR
    MLIRLLVMDialect
    MLIROptLib
    MLIRTransforms
    MLIRSCFToControlFlow
    MLIRAffineToStandard

    LLVMSupport

    CIRCTHW
    CIRCTHWTransforms
    CIRCTSeq
    CIRCTSeqTransforms
    CIRCTTransforms
    CIRCTImportVerilog
    CIRCTComb
    CIRCTLLHD
    CIRCTLLHDTransforms
    CIRCTMoore
    CIRCTMooreToCore
    CIRCTMooreTransforms
    CIRCTSupport
    CIRCTVerif

    EquivFusionSolverRunner
    EquivFusionToolEquivMiter
    EquivFusionWrite 
    FuncToHWModule
    EquivFusionRemoveRedundantFunc
    EquivFusionConvertCaseToLogicalComparison

    EquivFusionPassTemporalUnroll

    EquivFusionManager
    EquivFusionUtils
)

llvm_update_compile_flags(base)
