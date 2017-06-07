#include <climits>

#include "benchmark/benchmark_api.h"
#include "chai/ArrayManager.hpp"

void benchmark_arraymanager_alloc_default(benchmark::State& state) {
  chai::ArrayManager* manager = chai::ArrayManager::getInstance();

  while (state.KeepRunning()) {
    void* ptr = manager->allocate<char>(state.range_x());
    manager->free(ptr);
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

void benchmark_arraymanager_alloc_cpu(benchmark::State& state) {
  chai::ArrayManager* manager = chai::ArrayManager::getInstance();

  while (state.KeepRunning()) {
    void* ptr = manager->allocate<char>(state.range_x(), chai::CPU);
    manager->free(ptr);
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());

}

BENCHMARK(benchmark_arraymanager_alloc_default)->Range(1, INT_MAX);
BENCHMARK(benchmark_arraymanager_alloc_cpu)->Range(1, INT_MAX);

#if defined(CHAI_ENABLE_GPU)
void benchmark_arraymanager_alloc_gpu(benchmark::State& state) {
  chai::ArrayManager* manager = chai::ArrayManager::getInstance();

  while (state.KeepRunning()) {
    void* ptr = manager->allocate<char>(state.range_x(), chai::GPU);
    manager->free(ptr);
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}
BENCHMARK(benchmark_arraymanager_alloc_gpu)->Range(1, INT_MAX);
#endif

BENCHMARK_MAIN();
