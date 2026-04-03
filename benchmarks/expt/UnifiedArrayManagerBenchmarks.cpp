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

  static void UnifiedArrayManager_Data(benchmark::State& state,
                                       bool initialize_state,
                                       Context initial_context,
                                       bool initial_touch,
                                       Context call_context,
                                       bool call_touch,
                                       bool device_synchronized_before_call)
  {
    constexpr std::size_t size = 1;

    for (auto _ : state)
    {
      state.PauseTiming();

      auto& contextManager = ContextManager::getInstance();
      contextManager.reset();

      UnifiedArrayManager<int> manager{size};

      if (initialize_state)
      {
        ContextGuard guard{initial_context};
        int* initial_data = manager.data(/*touch=*/initial_touch);
        benchmark::DoNotOptimize(initial_data);
      }

      {
        ContextGuard guard{call_context};
        contextManager.setDeviceSynchronized(device_synchronized_before_call);
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

  static void UnifiedArrayManager_FirstData_HostNoTouch(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/false,
                             /*initial_context=*/Context::NONE,
                             /*initial_touch=*/false,
                             /*call_context=*/Context::HOST,
                             /*call_touch=*/false,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_FirstData_HostNoTouch)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_FirstData_Host(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/false,
                             /*initial_context=*/Context::NONE,
                             /*initial_touch=*/false,
                             /*call_context=*/Context::HOST,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_FirstData_Host)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_FirstData_DeviceNoTouch(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/false,
                             /*initial_context=*/Context::NONE,
                             /*initial_touch=*/false,
                             /*call_context=*/Context::DEVICE,
                             /*call_touch=*/false,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_FirstData_DeviceNoTouch)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_FirstData_Device(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/false,
                             /*initial_context=*/Context::NONE,
                             /*initial_touch=*/false,
                             /*call_context=*/Context::DEVICE,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_FirstData_Device)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_DataAfterHostTouch_HostNoTouch(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/true,
                             /*initial_context=*/Context::HOST,
                             /*initial_touch=*/true,
                             /*call_context=*/Context::HOST,
                             /*call_touch=*/false,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_DataAfterHostTouch_HostNoTouch)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_DataAfterHostTouch_Host(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/true,
                             /*initial_context=*/Context::HOST,
                             /*initial_touch=*/true,
                             /*call_context=*/Context::HOST,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_DataAfterHostTouch_Host)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_DataAfterHostTouch_Device(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/true,
                             /*initial_context=*/Context::HOST,
                             /*initial_touch=*/true,
                             /*call_context=*/Context::DEVICE,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_DataAfterHostTouch_Device)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_DataAfterDeviceTouch_HostSynchronized(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/true,
                             /*initial_context=*/Context::DEVICE,
                             /*initial_touch=*/true,
                             /*call_context=*/Context::HOST,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_DataAfterDeviceTouch_HostSynchronized)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_DataAfterDeviceTouch_HostUnsynchronized(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/true,
                             /*initial_context=*/Context::DEVICE,
                             /*initial_touch=*/true,
                             /*call_context=*/Context::HOST,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/false);
  }

  BENCHMARK(UnifiedArrayManager_DataAfterDeviceTouch_HostUnsynchronized)
    ->Unit(benchmark::kNanosecond);

  static void UnifiedArrayManager_DataAfterDeviceTouch_Device(benchmark::State& state)
  {
    UnifiedArrayManager_Data(state,
                             /*initialize_state=*/true,
                             /*initial_context=*/Context::DEVICE,
                             /*initial_touch=*/true,
                             /*call_context=*/Context::DEVICE,
                             /*call_touch=*/true,
                             /*device_synchronized_before_call=*/true);
  }

  BENCHMARK(UnifiedArrayManager_DataAfterDeviceTouch_Device)
    ->Unit(benchmark::kNanosecond);
}  // namespace

BENCHMARK_MAIN();
