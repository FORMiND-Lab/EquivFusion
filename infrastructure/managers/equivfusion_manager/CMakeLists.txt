aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(EquivFusionManager STATIC ${SRC_FILES})

target_include_directories(EquivFusionManager PUBLIC ${INCLUDE_DIRS})

target_link_libraries(EquivFusionManager
    PUBLIC

    MLIRFuncInlinerExtension
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRAffineDialect
    MLIRIR
    MLIRSMT
    

    CIRCTHW
    CIRCTSeq
    CIRCTComb
    CIRCTLLHD
    CIRCTMoore
    CIRCTVerif
    CIRCTEmit
    CIRCTOM
)

llvm_update_compile_flags(EquivFusionManager)

