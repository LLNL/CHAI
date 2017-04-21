#include "benchmark/benchmark_api.h"

#include "chai/ManagedArray.hpp"

void benchmark_managedarray_alloc_default(benchmark::State& state) {
    while (state.KeepRunning()) {
        for (int i=0; i < state.range_x(); ++i) {
          benchmark::DoNotOptimize(chai::ManagedArray<double>(state.range_x()));
      }
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

void benchmark_managedarray_alloc_cpu(benchmark::State& state) {
    while (state.KeepRunning()) {
        for (int i=0; i < state.range_x(); ++i) {
          benchmark::DoNotOptimize(chai::ManagedArray<double>(state.range_x(), chai::CPU));
      }
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());

}

void benchmark_managedarray_alloc_gpu(benchmark::State& state) {
    while (state.KeepRunning()) {
        for (int i=0; i < state.range_x(); ++i) {
          benchmark::DoNotOptimize(chai::ManagedArray<double>(state.range_x(), chai::GPU));
      }
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

BENCHMARK(benchmark_managedarray_alloc_default)->Range(8, 8<<12)
BENCHMARK(benchmark_managedarray_alloc_cpu)->Range(8, 8<<12)
BENCHMARK(benchmark_managedarray_alloc_gpu)->Range(8, 8<<12)
