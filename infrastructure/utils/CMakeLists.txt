aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(utils STATIC ${SRC_FILES})

target_include_directories(utils PRIVATE ${INCLUDE_DIRS})
target_include_directories(utils PRIVATE ${READLINE_INCLUDE_DIRECTORY})

target_link_libraries(utils
    PUBLIC
    ${READLINE_LIBRARY}
    log                         # 显式链接log库
    -Wl,--whole-archive base -Wl,--no-whole-archive
)


