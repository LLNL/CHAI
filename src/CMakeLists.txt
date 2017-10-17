#######################################################################
# Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
# rights reserved.
# 
# Produced at the Lawrence Livermore National Laboratory.
# 
# This file is part of CHAI.
# 
# LLNL-CODE-705877
# 
# For details, see https:://github.com/LLNL/CHAI
# Please also see the NOTICE and LICENSE files.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the
#   distribution.
# 
# - Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#######################################################################

set(CHAI_ENABLE_CNMEM ${ENABLE_CNMEM})
set(CHAI_ENABLE_PICK_SET_INCR_DECR ${ENABLE_PICK_SET_INCR_DECR})
set(CHAI_ENABLE_CUDA ${ENABLE_CUDA})
set(CHAI_ENABLE_IMPLICIT_CONVERSIONS ${ENABLE_IMPLICIT_CONVERSIONS})
set(CHAI_DISABLE_RM ${DISABLE_RM})
set(CHAI_ENABLE_UM ${ENABLE_UM})

configure_file(
  ${PROJECT_SOURCE_DIR}/src/config.hpp.in
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

set (chai_depends)

if (ENABLE_CUDA)
  set (chai_depends
    cuda_runtime)
endif ()
if (ENABLE_CNMEM)
  set (chai_depends
    ${chai_depends}
    cnmem)
endif()

message(STATUS ${chai_headers})

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  DEPENDS_ON ${chai_depends})

install(FILES util/forall.hpp DESTINATION include/chai/util/)

target_include_directories(
  chai
  PRIVATE ${PROJECT_BINARY_DIR}/include)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/chai-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake
  DESTINATION share/chai/cmake/)

install(
  FILES ${PROJECT_BINARY_DIR}/include/chai/config.hpp
  DESTINATION include/chai)

install(
  TARGETS chai
  EXPORT chai-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(EXPORT chai-targets DESTINATION share/chai/cmake/)

add_subdirectory(tests)
if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_subdirectory(examples)
