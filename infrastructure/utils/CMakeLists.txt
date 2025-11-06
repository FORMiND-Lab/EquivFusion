aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/log LOG_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/path PATH_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/string STRING_FILES)

add_library(EquivFusionUtils STATIC
    ${LOG_FILES}
    ${PATH_FILES}
    ${STRING_FILES}
)



