aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(base OBJECT ${SRC_FILES})

target_include_directories(base PRIVATE ${INCLUDE_DIRS})
target_include_directories(base PRIVATE ${READLINE_INCLUDE_ABSOLUTE_DIRECTORY})

target_link_libraries(base PUBLIC log)
target_link_libraries(base PUBLIC ${READLINE_LIBRARY})
