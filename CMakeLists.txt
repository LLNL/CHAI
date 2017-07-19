cmake_minimum_required(VERSION 3.2)

project(Chai LANGUAGES CXX)

set(ENABLE_CUDA On CACHE BOOL "Enable CUDA")
set(ENABLE_OPENMP On CACHE BOOL "Enable OpenMP")
set(ENABLE_BENCHMARKS On CACHE Bool "Enable benchmarks")
set(ENABLE_COPY_HEADERS On CACHE BOOL "")
set(BLT_CXX_STD c++11 CACHE STRING "")

option(ENABLE_CNMEM "Enable cnemem for GPU allocation" Off)
option(ENABLE_THIN_UM "Make ManagedArray a thin wrapper around CUDA Unified Memory" On)
mark_as_advanced(ENABLE_THIN_UM)

################################
# BLT
################################
if (NOT EXISTS ${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)
  message(FATAL_ERROR "\
  The BLT submodule is not present. \
  If in git repository run the following two commands:\n \
  git submodule init\n \
  git submodule update")
endif()

include(blt/SetupBLT.cmake)

include(cmake/ChaiBasics.cmake)

add_subdirectory(src)
add_subdirectory(docs)
