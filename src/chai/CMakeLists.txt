##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set(CHAI_ENABLE_PICK ${ENABLE_PICK})
set(CHAI_ENABLE_CUDA ${ENABLE_CUDA})
set(CHAI_ENABLE_HIP ${ENABLE_HIP})
set(CHAI_ENABLE_IMPLICIT_CONVERSIONS ${ENABLE_IMPLICIT_CONVERSIONS})
set(CHAI_DISABLE_RM ${DISABLE_RM})
set(CHAI_ENABLE_UM ${ENABLE_UM})
set(CHAI_ENABLE_RAJA_PLUGIN ${ENABLE_RAJA_PLUGIN})

configure_file(
  ${PROJECT_SOURCE_DIR}/src/chai/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/chai/config.hpp)

set (chai_headers
  ArrayManager.hpp
  ArrayManager.inl
  ChaiMacros.hpp
  ExecutionSpaces.hpp
  ManagedArray.hpp
  ManagedArray.inl
  PointerRecord.hpp
  Types.hpp)

if(DISABLE_RM)
  set(chai_headers
    ${chai_headers}
    ManagedArray_thin.inl)
endif ()

set (chai_sources
  ArrayManager.cpp)

set (chai_depends
  umpire)

if (ENABLE_CUDA)
  set (chai_depends
    ${chai_depends}
    cuda_runtime)
endif ()
if (ENABLE_HIP)
  set (chai_depends
    ${chai_depends}
    hip_runtime)
endif ()

if (ENABLE_RAJA_PLUGIN)
  set (chai_headers
    ${chai_headers}
    pluginLinker.hpp
    ManagedArrayView.hpp
    RajaExecutionSpacePlugin.hpp)

  set (chai_sources
    ${chai_sources}
    RajaExecutionSpacePlugin.cpp)

  set (chai_depends
    ${chai_depends}
    RAJA)
endif ()

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  DEPENDS_ON ${chai_depends})

install(FILES ${chai_headers} DESTINATION include/chai/)

install(FILES ../util/forall.hpp DESTINATION include/chai/util/)

target_include_directories(
  chai
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/chai/chai-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake
  DESTINATION share/chai/cmake/)

install(
  FILES ${PROJECT_BINARY_DIR}/include/chai/config.hpp
  DESTINATION include/chai)

install(
  FILES ${chai_headers}
  DESTINATION include/chai)

install(
  TARGETS chai
  EXPORT chai-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(EXPORT chai-targets DESTINATION share/chai/cmake/)
