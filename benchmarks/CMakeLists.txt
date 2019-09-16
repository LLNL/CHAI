#######################################################################
# Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
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

set (chai_benchmark_depends
  chai
  gbenchmark)

if (ENABLE_CUDA)
  set (chai_benchmark_depends
    ${chai_benchmark_depends}
    cuda)
endif ()
if (ENABLE_HIP)
  set (chai_benchmark_depends
    ${chai_benchmark_depends}
    hip)
endif ()

blt_add_executable(
  NAME arraymanager_benchmarks
  SOURCES chai_arraymanager_benchmarks.cpp
  DEPENDS_ON ${chai_benchmark_depends})

blt_add_benchmark(
  NAME arraymanager_benchmarks
  COMMAND arraymanager_benchmarks)

blt_add_executable(
  NAME managedarray_benchmarks
  SOURCES chai_managedarray_benchmarks.cpp
  DEPENDS_ON ${chai_benchmark_depends})

blt_add_benchmark(
  NAME managedarray_benchmarks
  COMMAND managedarray_benchmarks)
