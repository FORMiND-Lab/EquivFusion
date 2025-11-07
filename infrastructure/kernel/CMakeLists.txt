aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_FILES)

add_library(EquivFusionKernel STATIC ${SRC_FILES})

target_link_libraries(EquivFusionKernel
    PUBLIC
    ${READLINE_LIBRARY}
    EquivFusionUtils
    -Wl,--whole-archive base -Wl,--no-whole-archive
)


