aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(EquivFusionKernal STATIC ${SRC_FILES})

target_link_libraries(EquivFusionKernal
    PUBLIC
    ${READLINE_LIBRARY}
    EquivFusionUtils
    -Wl,--whole-archive base -Wl,--no-whole-archive
)


