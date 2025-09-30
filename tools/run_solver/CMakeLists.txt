aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)
add_executable(run_solver ${SRC_FILES})

target_include_directories(run_solver PRIVATE ${INCLUDE_DIRS})
target_link_libraries(run_solver PUBLIC solving log)
