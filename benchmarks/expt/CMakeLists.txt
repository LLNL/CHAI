##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(chai_expt_benchmark_depends
  chai
  gbenchmark)

if (CHAI_ENABLE_CUDA)
  set(chai_expt_benchmark_depends
    ${chai_expt_benchmark_depends}
    cuda)
endif ()

if (CHAI_ENABLE_HIP)
  set(chai_expt_benchmark_depends
    ${chai_expt_benchmark_depends}
    blt::hip)
endif ()

if (CHAI_ENABLE_CUDA OR CHAI_ENABLE_HIP)
  blt_add_executable(
    NAME DualArrayManagerBenchmarks
    SOURCES DualArrayManagerBenchmarks.cpp
    DEPENDS_ON ${chai_expt_benchmark_depends})

  blt_add_benchmark(
    NAME DualArrayManagerBenchmarks
    COMMAND DualArrayManagerBenchmarks)

  blt_add_executable(
    NAME UnifiedArrayManagerBenchmarks
    SOURCES UnifiedArrayManagerBenchmarks.cpp
    DEPENDS_ON ${chai_expt_benchmark_depends})

  blt_add_benchmark(
    NAME UnifiedArrayManagerBenchmarks
    COMMAND UnifiedArrayManagerBenchmarks)
endif ()
