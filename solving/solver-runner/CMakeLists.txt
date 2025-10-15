aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)
add_library(EquivFusionSolverRunner STATIC ${SRC_FILES})
target_include_directories(EquivFusionSolverRunner PRIVATE ${INCLUDE_DIRS})
