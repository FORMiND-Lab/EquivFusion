aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)
add_executable(equiv_fusion ${SRC_FILES})

target_include_directories(equiv_fusion PRIVATE ${INCLUDE_DIRS})
target_link_libraries(equiv_fusion PUBLIC utils)
