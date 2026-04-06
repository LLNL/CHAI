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
  using ::chai::expt::ContextManager;
  using ::chai::expt::UnifiedArrayManager;

  static void UnifiedArrayManager_DataSequence(benchmark::State& state,
                                               Context initial_context,
                                               bool initial_touch,
                                               Context call_context,
                                               bool call_touch)
  {
    constexpr std::size_t size = 1;

    for (auto _ : state)
    {
      state.PauseTiming();

      auto& contextManager = ContextManager::getInstance();
      contextManager.reset();

      UnifiedArrayManager<int> manager{size};
      {
        ContextGuard guard{initial_context};
        int* initial_data = manager.data(/*touch=*/initial_touch);
        benchmark::DoNotOptimize(initial_data);
      }

      {
        ContextGuard guard{call_context};
        state.ResumeTiming();  // measure only the data() call
        int* data = manager.data(/*touch=*/call_touch);
        benchmark::DoNotOptimize(data);
        state.PauseTiming();  // exclude guard destruction
      }
    }

    state.SetItemsProcessed(state.iterations());
  }

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
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20);

  static void UnifiedArrayManager_AfterHostRead_DataHostRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::HOST,
                                     /*initial_touch=*/false,
                                     /*call_context=*/Context::HOST,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterHostRead_DataHostRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterHostWrite_DataHostRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::HOST,
                                     /*initial_touch=*/true,
                                     /*call_context=*/Context::HOST,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterHostWrite_DataHostRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterHostRead_DataDeviceRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::HOST,
                                     /*initial_touch=*/false,
                                     /*call_context=*/Context::DEVICE,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterHostRead_DataDeviceRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterHostWrite_DataDeviceRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::HOST,
                                     /*initial_touch=*/true,
                                     /*call_context=*/Context::DEVICE,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterHostWrite_DataDeviceRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterDeviceRead_DataHostRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::DEVICE,
                                     /*initial_touch=*/false,
                                     /*call_context=*/Context::HOST,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterDeviceRead_DataHostRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterDeviceWrite_DataHostRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::DEVICE,
                                     /*initial_touch=*/true,
                                     /*call_context=*/Context::HOST,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterDeviceWrite_DataHostRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterDeviceRead_DataDeviceRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::DEVICE,
                                     /*initial_touch=*/false,
                                     /*call_context=*/Context::DEVICE,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterDeviceRead_DataDeviceRead)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_AfterDeviceWrite_DataDeviceRead(benchmark::State& state)
  {
    UnifiedArrayManager_DataSequence(state,
                                     /*initial_context=*/Context::DEVICE,
                                     /*initial_touch=*/true,
                                     /*call_context=*/Context::DEVICE,
                                     /*call_touch=*/false);
  }

  BENCHMARK(UnifiedArrayManager_AfterDeviceWrite_DataDeviceRead)
    ->Unit(benchmark::kNanosecond);
}  // namespace

BENCHMARK_MAIN();
