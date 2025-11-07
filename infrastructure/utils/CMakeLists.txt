aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/log-util LOG_UTIL_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/path-util PATH_UTIL_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/string-util STRING_UTIL_FILES)

add_library(EquivFusionUtils STATIC
    ${LOG_UTIL_FILES}
    ${PATH_UTIL_FILES}
    ${STRING_UTIL_FILES}
)



