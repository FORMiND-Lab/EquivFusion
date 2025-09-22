cmake_minimum_required(VERSION 3.20)

project(equiv_fusion)

set(CMAKE_CXX_STANDARD 20)

set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
set(OUTPUT_LIB_DIR ${CMAKE_BINARY_DIR}/lib/)
set(OUTPUT_BIN_DIR ${CMAKE_BINARY_DIR}/bin/)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_LIB_DIR})

# -------------------------------------------
# Readline Library Configuration
# -------------------------------------------

set(READLINE_INCLUDE_DIRECTORY "" CACHE STRING "Absolute path to readline include directory")
set(READLINE_LIBRARY_PATH "" CACHE STRING "Absolute path to readline library ")

if (READLINE_INCLUDE_DIRECTORY) 
    if (NOT EXISTS ${READLINE_INCLUDE_DIRECTORY}) 
        message(FATAL_ERROR "'${READLINE_INCLUDE_DIRECTORY}' does not exist")
    endif()
    if (NOT IS_DIRECTORY ${READLINE_INCLUDE_DIRECTORY}) 
        message(FATAL_ERROR "'${READLINE_INCLUDE_DIRECTORY}' is not a directory")
    endif()
endif()

if (READLINE_LIBRARY_PATH) 
    if (NOT EXISTS ${READLINE_LIBRARY_PATH}) 
        message(FATAL_ERROR "'${READLINE_LIBRARY_PATH}' does not exist")
    endif()

    set(READLINE_LIBRARY ${READLINE_LIBRARY_PATH})
else() 
    find_library(READLINE_LIBRARY NAMES readline)
    if (NOT READLINE_LIBRARY) 
        message(FATAL_ERROR "readline library not found")
    endif()
endif()

# -------------------------------------------
# CIRCT Configuration
# -------------------------------------------

set(CIRCT_LINUX_DOWNLOAD_URL "https://github.com/llvm/circt/releases/download/firtool-1.126.0/circt-full-shared-linux-x64.tar.gz")
set(CIRCT_APPLE_DOWNLOAD_URL "https://github.com/llvm/circt/releases/download/firtool-1.126.0/circt-full-shared-macos-x64.tar.gz")
set(CIRCT_PREBUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/circt_prebuild")
set(CIRCT_TAR_PATH "${CIRCT_PREBUILD_DIR}/circt_prebuild.tar.gz")

## Specify a CIRCT built locally instead of downloading a prebuild CIRCT package.
set(CIRCT_ROOT_DIR "" CACHE STRING "Absolute path to local circt.")
set(LLVM_ROOT_DIR "" CACHE STRING "Absolute path to local LLVM.")
option(CIRCT_REDOWNLOAD "Redownload CIRCT" OFF)

if (CIRCT_ROOT_DIR) 
    if (NOT EXISTS ${CIRCT_ROOT_DIR})
        message(FATAL_ERROR "'${CIRCT_ROOT_DIR}' does not exist")
    endif()

    if (LLVM_ROOT_DIR) 
        if (NOT EXISTS ${LLVM_ROOT_DIR})
            message(FATAL_ERROR "'${LLVM_ROOT_DIR}' does not exist")
        endif()

        set(LLVM_DIR ${LLVM_ROOT_DIR}/lib/cmake/llvm)
        set(MLIR_DIR ${LLVM_ROOT_DIR}/lib/cmake/mlir)
    else()
        message(FATAL_ERROR "Local LLVM used by local CIRCT must be specified by 'LLVM_ROOT_DIR' if a local CIRCT is specified")
    endif()

    set(CIRCT_PACKAGE_DIR ${CIRCT_ROOT_DIR}) 
else() 
    if(CIRCT_REDOWNLOAD AND EXISTS ${CIRCT_PREBUILD_DIR})
        execute_process(COMMAND rm -rf ${CIRCT_PREBUILD_DIR})
    endif()

    if (NOT EXISTS ${CIRCT_PREBUILD_DIR}) 
        execute_process(COMMAND mkdir -p ${CIRCT_PREBUILD_DIR})
        message(STATUS "Downloading prebuild CIRCT package to '${CIRCT_PREBUILD_DIR}'")

        if (APPLE) 
            set(CIRCT_DOWNLOAD_URL ${CIRCT_APPLE_DOWNLOAD_URL})
        elseif (UNIX)
            set(CIRCT_DOWNLOAD_URL ${CIRCT_LINUX_DOWNLOAD_URL})
        endif()

        execute_process(COMMAND curl -L ${CIRCT_DOWNLOAD_URL} -o ${CIRCT_TAR_PATH} RESULT_VARIABLE DOWNLOAD_RESULT)
        if (NOT DOWNLOAD_RESULT EQUAL 0) 
            message(FATAL_ERROR "Failed to download prebuild CIRCT package")
        endif()

        execute_process(COMMAND tar -xzf ${CIRCT_TAR_PATH} --strip-components=1 -C ${CIRCT_PREBUILD_DIR} RESULT_VARIABLE EXTRACT_RESULT)
        if (NOT EXTRACT_RESULT EQUAL 0) 
            message(FATAL_ERROR "Failed to extract prebuild CIRCT package")
        endif()

        execute_process(COMMAND rm -rf ${CIRCT_TAR_PATH})
        set(CIRCT_PACKAGE_DIR ${CIRCT_PREBUILD_DIR})
    else()
        message(STATUS "'${CIRCT_PREBUILD_DIR}' already exists, skip downloading")
        set(CIRCT_PACKAGE_DIR ${CIRCT_PREBUILD_DIR})
    endif()
endif()

message(STATUS "Using CIRCT in '${CIRCT_PACKAGE_DIR}'")

set(CIRCT_DIR ${CIRCT_PACKAGE_DIR}/lib/cmake/circt)

find_package(CIRCT REQUIRED CONFIG)
message(STATUS "Using CIRCTConfig.cmake in '${CIRCT_DIR}'")

list(APPEND CMAKE_MODULE_PATH ${CIRCT_CMAKE_DIR})
include(AddCIRCT)

include_directories(${CIRCT_INCLUDE_DIRS})


# -------------------------------------------
# MLIR/LLVM Configuration
# -------------------------------------------

# Specify a local LLVM used by local CIRCT, only effective when local CIRCT is specified.

message(STATUS "Using LLVMConfig.cmake in '${LLVM_DIR}'")
message(STATUS "Using MLIRConfig.cmake in '${MLIR_DIR}'")

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# -------------------------------------------
# Add Subdirectories
# -------------------------------------------

add_subdirectory(infrastructure)
add_subdirectory(client)
add_subdirectory(tools)
add_subdirectory(solving)






