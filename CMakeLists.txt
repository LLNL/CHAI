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

cmake_minimum_required(VERSION 3.2)

project(Chai LANGUAGES CXX)

set(ENABLE_CUDA On CACHE BOOL "Enable CUDA")
set(ENABLE_OPENMP On CACHE BOOL "Enable OpenMP")
set(ENABLE_BENCHMARKS On CACHE Bool "Enable benchmarks")
option(ENABLE_CNMEM "Enable cnemem for GPU allocation" Off)
option(ENABLE_IMPLICIT_CONVERSIONS "Enable implicit conversions to-from raw pointers" On)
option(ENABLE_UM "Enable Unified Memory" On)
option(ENABLE_THIN_UM "Make ManagedArray a thin wrapper around CUDA Unified Memory" Off)
mark_as_advanced(ENABLE_THIN_UM)

if (ENABLE_THIN_UM AND NOT ENABLE_CUDA)
  message(FATAL_ERROR "Option ENABLE_THIN_UM requires ENABLE_CUDA")
endif()

if (ENABLE_UM AND NOT ENABLE_CUDA)
  message(FATAL_ERROR "Option ENABLE_UM requires ENABLE_CUDA")
endif()

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
