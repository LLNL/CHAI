#include <climits>

#include "benchmark/benchmark_api.h"

#include "chai/ManagedArray.hpp"
#include "../util/forall.hpp"

void benchmark_managedarray_alloc_default(benchmark::State& state) {
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range_x());
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

void benchmark_managedarray_alloc_cpu(benchmark::State& state) {
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range_x(), chai::CPU);
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

void benchmark_managedarray_alloc_gpu(benchmark::State& state) {
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range_x(), chai::GPU);
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

BENCHMARK(benchmark_managedarray_alloc_default)->Range(1, INT_MAX);
BENCHMARK(benchmark_managedarray_alloc_cpu)->Range(1, INT_MAX);
BENCHMARK(benchmark_managedarray_alloc_gpu)->Range(1, INT_MAX);

void benchmark_managedarray_move(benchmark::State& state)
{
  chai::ManagedArray<char> array(state.range_x());

  forall(sequential(), 0, 1, [=] (int i) {
      array[i] = 'b';
  });

  /*
   * Move data back and forth between CPU and GPU.
   *
   * Kernels just touch the data, but are still included in timing.
   */
  while (state.KeepRunning()) {
    forall(cuda(), 0, 1, [=] __device__ (int i) {
        array[i] = 'a';
    });

    forall(sequential(), 0, 1, [=] (int i) {
        array[i] = 'b';
    });
  }

  array.free();
}

BENCHMARK(benchmark_managedarray_move)->Range(1, INT_MAX);

BENCHMARK_MAIN();
