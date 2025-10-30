add_library(EquivFusionRead
    read_base.cpp
    read_mlir.cpp
)

target_link_libraries(EquivFusionRead
    PRIVATE
  
    MLIRParser
    
    log
)
