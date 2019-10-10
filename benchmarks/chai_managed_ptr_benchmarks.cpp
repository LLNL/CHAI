// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
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
#include "chai/managed_ptr.hpp"

#include "../src/util/forall.hpp"

class Base {
   public:
      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class Derived : public Base {
   public:
      CHAI_HOST_DEVICE Derived(int value) : Base(), m_value(value) {}

      CHAI_HOST_DEVICE int getValue() const override { return m_value; }

   private:
      int m_value = -1;
};

void benchmark_managed_ptr_construction_and_destruction(benchmark::State& state)
{
  while (state.KeepRunning()) {
    chai::managed_ptr<Base> temp = chai::make_managed<Derived>(state.range(0));
    temp.free();
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(benchmark_managed_ptr_construction_and_destruction)->Range(1, 1);

static chai::managed_ptr<Base> helper1 = chai::make_managed<Derived>(1);

void benchmark_managed_ptr_use_cpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    auto helper = helper1;
    forall(sequential(), 0, 1, [=] (int i) { (void) helper->getValue(); });
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(benchmark_managed_ptr_use_cpu)->Range(1, 1);

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

static chai::managed_ptr<Base> helper2 = chai::make_managed<Derived>(2);

void benchmark_managed_ptr_use_gpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    auto helper = helper2;
    forall(gpu(), 0, 1, [=] __device__ (int i) { (void) helper->getValue(); });
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(benchmark_managed_ptr_use_gpu)->Range(1, 1);

#endif

BENCHMARK_MAIN();
