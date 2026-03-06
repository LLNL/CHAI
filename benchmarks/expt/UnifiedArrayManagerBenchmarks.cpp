//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include <cstddef>

#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/UnifiedArrayManager.hpp"

namespace {
  using ::chai::expt::Context;
  using ::chai::expt::ContextGuard;
  using ::chai::expt::UnifiedArrayManager;

  static void UnifiedArrayManager_DefaultConstruct(benchmark::State& state)
  {
    for (auto _ : state)
    {
      UnifiedArrayManager<int> manager{};
      benchmark::DoNotOptimize(manager);
    }
  }

  BENCHMARK(UnifiedArrayManager_DefaultConstruct);

  static void UnifiedArrayManager_SizeConstruct(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      UnifiedArrayManager<int> manager{size};
      benchmark::DoNotOptimize(manager);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(int));
  }

  BENCHMARK(UnifiedArrayManager_SizeConstruct)
    ->Arg(0)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 20);

  static void UnifiedArrayManager_FirstData_HostConst(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      state.PauseTiming();
      UnifiedArrayManager<int> manager{size};
      state.ResumeTiming();

      {
        ContextGuard guard{Context::HOST};
        int* data = manager.data(/*touch=*/false);
        benchmark::DoNotOptimize(data);
      }
    }

    state.SetItemsProcessed(state.iterations());
  }

  BENCHMARK(UnifiedArrayManager_FirstData_HostConst)
    ->Arg(0)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 20);

  static void UnifiedArrayManager_FirstData_Host(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      state.PauseTiming();
      UnifiedArrayManager<int> manager{size};
      state.ResumeTiming();

      {
        ContextGuard guard{Context::HOST};
        int* data = manager.data(/*touch=*/true);
        benchmark::DoNotOptimize(data);
      }
    }

    state.SetItemsProcessed(state.iterations());
  }

  BENCHMARK(UnifiedArrayManager_FirstData_Host)
    ->Arg(0)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 20);

  static void UnifiedArrayManager_FirstData_DeviceConst(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      state.PauseTiming();
      UnifiedArrayManager<int> manager{size};
      state.ResumeTiming();

      {
        ContextGuard guard{Context::DEVICE};
        int* data = manager.data(/*touch=*/false);
        benchmark::DoNotOptimize(data);
      }
    }

    state.SetItemsProcessed(state.iterations());
  }

  BENCHMARK(UnifiedArrayManager_FirstData_DeviceConst)
    ->Arg(0)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 20);

  static void UnifiedArrayManager_FirstData_Device(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      state.PauseTiming();
      UnifiedArrayManager<int> manager{size};
      state.ResumeTiming();

      {
        ContextGuard guard{Context::DEVICE};
        int* data = manager.data(/*touch=*/true);
        benchmark::DoNotOptimize(data);
      }
    }

    state.SetItemsProcessed(state.iterations());
  }

  BENCHMARK(UnifiedArrayManager_FirstData_Device)
    ->Arg(0)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 20);
}  // namespace

BENCHMARK_MAIN();
