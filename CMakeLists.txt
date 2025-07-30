cmake_minimum_required(VERSION 3.20)

project(equiv_fusion)

set(CMAKE_CXX_STANDARD 20)

set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
set(OUTPUT_LIB_DIR ${CMAKE_BINARY_DIR}/lib/)
set(OUTPUT_BIN_DIR ${CMAKE_BINARY_DIR}/bin/)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})

set(READLINE_INCLUDE_ABSOLUTE_DIRECTORY "" CACHE STRING "Absolute path to readline include directory")
set(READLINE_LIBRARY_ABSOLUTE_PATH "" CACHE STRING "Absolute path to readline library ")

if (READLINE_INCLUDE_ABSOLUTE_DIRECTORY) 
    if (NOT EXISTS ${READLINE_INCLUDE_ABSOLUTE_DIRECTORY}) 
        message(FATAL_ERROR "'${READLINE_INCLUDE_ABSOLUTE_DIRECTORY}' does not exist")
    endif()
    if (NOT IS_DIRECTORY ${READLINE_INCLUDE_ABSOLUTE_DIRECTORY}) 
        message(FATAL_ERROR "'${READLINE_INCLUDE_ABSOLUTE_DIRECTORY}' is not a directory")
    endif()
endif()

if (READLINE_LIBRARY_ABSOLUTE_PATH) 
    if (NOT EXISTS ${READLINE_LIBRARY_ABSOLUTE_PATH}) 
        message(FATAL_ERROR "'${READLINE_LIBRARY_ABSOLUTE_PATH}' does not exist")
    endif()

    set(READLINE_LIBRARY ${READLINE_LIBRARY_ABSOLUTE_PATH})
else() 
    find_library(READLINE_LIBRARY NAMES readline)
    if (NOT READLINE_LIBRARY) 
        message(FATAL_ERROR "readline library not found")
    endif()
endif()

add_subdirectory(infrastructure)
add_subdirectory(client)






