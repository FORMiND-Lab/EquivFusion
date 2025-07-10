aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(log STATIC ${SRC_FILES})

target_include_directories(log PRIVATE ${INCLUDE_DIRS})
