cmake_minimum_required(VERSION 3.2)

project(Chai LANGUAGES CXX)

set(ENABLE_CUDA On CACHE BOOL "Enable CUDA")
set(ENABLE_OPENMP On CACHE BOOL "Enable OpenMP")
set(ENABLE_BENCHMARKS On CACHE Bool "Enable benchmarks")
option(ENABLE_CNMEM "Enable cnemem for GPU allocation" Off)
option(ENABLE_IMPLICIT_CONVERSIONS "Enable implicit conversions to-from raw pointers" On)
option(ENABLE_THIN_UM "Make ManagedArray a thin wrapper around CUDA Unified Memory" Off)
mark_as_advanced(ENABLE_THIN_UM)

set(ENABLE_COPY_HEADERS On CACHE BOOL "")
set(BLT_CXX_STD c++11 CACHE STRING "")


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
