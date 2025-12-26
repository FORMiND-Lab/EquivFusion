add_circt_library(EquivFusionToolReadFIRRTL
    read_firrtl.cpp

    LINK_LIBS PUBLIC
    CIRCTImportFIRFile
    CIRCTFirtool

    EquivFusionManager
    EquivFusionUtils
)
