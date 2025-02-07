//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include <climits>

#include "benchmark/benchmark.h"

#include "chai/ArrayManager.hpp"

// void benchmark_arraymanager_alloc_default(benchmark::State& state) {
//   chai::ArrayManager* manager = chai::ArrayManager::getInstance();
//
//   while (state.KeepRunning()) {
//     void* ptr = manager->allocate<char>(state.range(0));
//     manager->free(ptr);
//   }
//
//   state.SetItemsProcessed(state.iterations() * state.range(0));
// }
//
// void benchmark_arraymanager_alloc_cpu(benchmark::State& state) {
//   chai::ArrayManager* manager = chai::ArrayManager::getInstance();
//
//   while (state.KeepRunning()) {
//     void* ptr = manager->allocate<char>(state.range(0), chai::CPU);
//     manager->free(ptr);
//   }
//
//   state.SetItemsProcessed(state.iterations() * state.range(0));
//
// }
//
// BENCHMARK(benchmark_arraymanager_alloc_default)->Range(1, INT_MAX);
// BENCHMARK(benchmark_arraymanager_alloc_cpu)->Range(1, INT_MAX);
//
// #if defined(CHAI_ENABLE_CUDA)
// void benchmark_arraymanager_alloc_gpu(benchmark::State& state) {
//   chai::ArrayManager* manager = chai::ArrayManager::getInstance();
//
//   while (state.KeepRunning()) {
//     void* ptr = manager->allocate<char>(state.range(0), chai::GPU);
//     manager->free(ptr);
//   }
//
//   state.SetItemsProcessed(state.iterations() * state.range(0));
// }
// BENCHMARK(benchmark_arraymanager_alloc_gpu)->Range(1, INT_MAX);
// #endif
//
BENCHMARK_MAIN();
