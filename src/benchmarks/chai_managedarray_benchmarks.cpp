// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
// rights reserved.
// 
// Produced at the Lawrence Livermore National Laboratory.
// 
// This file is part of CHAI.
// 
// LLNL-CODE-705877
// 
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
// 
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
#include <climits>

#include "benchmark/benchmark_api.h"

#include "chai/config.hpp"
#include "chai/ManagedArray.hpp"

#include "../util/forall.hpp"

void benchmark_managedarray_alloc_default(benchmark::State& state) {
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range(0));
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}

void benchmark_managedarray_alloc_cpu(benchmark::State& state) {
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range(0), chai::CPU);
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}

BENCHMARK(benchmark_managedarray_alloc_default)->Range(1, INT_MAX);
BENCHMARK(benchmark_managedarray_alloc_cpu)->Range(1, INT_MAX);

#if defined(CHAI_ENABLE_CUDA)
void benchmark_managedarray_alloc_gpu(benchmark::State& state) {
  while (state.KeepRunning()) {
    chai::ManagedArray<char> array(state.range(0), chai::GPU);
    array.free();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(benchmark_managedarray_alloc_gpu)->Range(1, INT_MAX);
#endif


#if defined(CHAI_ENABLE_CUDA)
void benchmark_managedarray_move(benchmark::State& state)
{
  chai::ManagedArray<char> array(state.range(0));

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
#endif

BENCHMARK_MAIN();
