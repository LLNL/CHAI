//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include <climits>

#include "benchmark/benchmark.h"

#include "chai/ManagedArray.hpp"
#include "chai/config.hpp"

#include "../src/util/forall.hpp"

void benchmark_managedarray_alloc_default(benchmark::State& state)
{
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range(0));
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}

void benchmark_managedarray_alloc_cpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range(0), chai::CPU);
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}

BENCHMARK(benchmark_managedarray_alloc_default)->Range(1, INT_MAX);
BENCHMARK(benchmark_managedarray_alloc_cpu)->Range(1, INT_MAX);

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
void benchmark_managedarray_alloc_gpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range(0), chai::GPU);
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(benchmark_managedarray_alloc_gpu)->Range(1, INT_MAX);
#endif


#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
void benchmark_managedarray_move(benchmark::State& state)
{
  chai::ManagedArray<char> array(state.range(0));

  forall(sequential(), 0, 1, [=](int i) { array[i] = 'b'; });

  /*
   * Move data back and forth between CPU and GPU.
   *
   * Kernels just touch the data, but are still included in timing.
   */
  while (state.KeepRunning()) {
    forall(gpu(), 0, 1, [=] __device__(int i) { array[i] = 'a'; });

    forall(sequential(), 0, 1, [=](int i) { array[i] = 'b'; });
  }

  array.free();
}

BENCHMARK(benchmark_managedarray_move)->Range(1, INT_MAX);
#endif

BENCHMARK_MAIN();
