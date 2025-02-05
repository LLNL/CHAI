//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_chai_benchmark_utils_HPP
#define CHAI_chai_benchmark_utils_HPP

#include "benchmark/benchmark_api.h"

static void malloc_ranges(benchmark::internal::Benchmark* b)
{
  // for (int i = 1; i <= (1<<30); i*=2)
  //      b->Args(i);
}

#endif  // CHAI_chai_benchmark_utils_HPP
