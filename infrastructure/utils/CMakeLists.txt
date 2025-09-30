aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(utils STATIC ${SRC_FILES})

target_include_directories(utils PRIVATE ${INCLUDE_DIRS})
target_include_directories(utils PRIVATE ${READLINE_INCLUDE_DIRECTORY})

target_link_libraries(utils PUBLIC ${READLINE_LIBRARY})
target_link_libraries(utils PUBLIC $<TARGET_OBJECTS:base>)
target_link_libraries(utils PUBLIC log)  # 显式链接log库
target_link_libraries(utils PUBLIC solving)


