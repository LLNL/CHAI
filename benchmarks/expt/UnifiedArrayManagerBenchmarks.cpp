//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include "chai/expt/UnifiedArrayManager.hpp"

namespace {
  using ::chai::expt::UnifiedArrayManager;

  static void unified_array_manager_default_construct(benchmark::State& state)
  {
    for (auto _ : state)
    {
      UnifiedArrayManager<int> manager{};
      benchmark::DoNotOptimize(manager);
    }
  }

  BENCHMARK(unified_array_manager_default_construct);
}  // namespace

BENCHMARK_MAIN();
