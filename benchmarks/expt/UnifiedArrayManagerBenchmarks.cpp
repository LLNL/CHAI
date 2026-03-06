//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include <cstddef>

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

  static void unified_array_manager_size_construct(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      UnifiedArrayManager<int> manager{size};
      benchmark::DoNotOptimize(manager);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(int));
  }

  BENCHMARK(unified_array_manager_size_construct)
    ->Arg(0)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 20);
}  // namespace

BENCHMARK_MAIN();
