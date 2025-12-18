aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_circt_library(EquivFusionMemrefHLS
    ${SRC_FILES}

    DEPENDS
    EquivFusionMemrefHLSIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRArithDialect
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRTransformUtils
    MLIRIR

    LLVMSupport

    CIRCTHW
    CIRCTComb
)
