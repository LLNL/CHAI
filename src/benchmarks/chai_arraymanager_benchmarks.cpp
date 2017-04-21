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

void benchmark_arraymanager_alloc_gpu(benchmark::State& state) {
  chai::ArrayManager* manager = chai::ArrayManager::getInstance();

  while (state.KeepRunning()) {
    void* ptr = manager->allocate<char>(state.range_x(), chai::GPU);
    manager->free(ptr);
  }

  state.SetItemsProcessed(state.iterations() * state.range_x());
}

void benchmark_arraymanager_move(benchmark::State& state)
{
  chai::ArrayManager* manager = chai::ArrayManager::getInstance();

  void* ptr = manager->allocate<char>(state.range_x());
  manager->setExecutionSpace(chai::CPU);
  manager->registerTouch(ptr);
  
  while (state.KeepRunning()) {
    manager->setExecutionSpace(chai::GPU);
    manager->move(ptr);
    manager->registerTouch(ptr);
    manager->setExecutionSpace(chai::CPU);
    manager->move(ptr);
    manager->registerTouch(ptr);
  }

  manager->free(ptr);
}

BENCHMARK(benchmark_arraymanager_alloc_default)->Range(1, INT_MAX);
BENCHMARK(benchmark_arraymanager_alloc_cpu)->Range(1, INT_MAX);
BENCHMARK(benchmark_arraymanager_alloc_gpu)->Range(1, INT_MAX);

BENCHMARK(benchmark_arraymanager_move)->Range(1, INT_MAX);

BENCHMARK_MAIN();
