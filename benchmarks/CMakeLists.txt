##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
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
