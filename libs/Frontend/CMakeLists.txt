add_library(EquivFusionFrontend
    frontend_base.cpp
    read_mlir.cpp
)

target_link_libraries(EquivFusionFrontend
    PRIVATE
  
    MLIRParser
    
    log
)
