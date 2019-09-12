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
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/DynamicPool.hpp"

#include "chai/ManagedArray.hpp"
#include "../src/util/forall.hpp"

#include <iostream>

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{

  auto& rm = umpire::ResourceManager::getInstance();

  auto cpu_pool =
      rm.makeAllocator<umpire::strategy::DynamicPool>("cpu_pool",
                                                      rm.getAllocator("HOST"));

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  auto gpu_pool =
      rm.makeAllocator<umpire::strategy::DynamicPool>("gpu_pool",
                                                      rm.getAllocator("DEVICE"));
#endif

  chai::ManagedArray<float> array(100, 
      std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
      , chai::GPU
#endif
      },
      std::initializer_list<umpire::Allocator>{cpu_pool
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
      , gpu_pool
#endif
      });

  forall(sequential(), 0, 100, [=](int i) { array[i] = 0.0f; });

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  forall(gpu(), 0, 100, [=] __device__(int i) { array[i] = 1.0f * i; });
#else
  forall(sequential(), 0, 100, [=] (int i) { array[i] = 1.0f * i; });
#endif

  forall(sequential(), 0, 100, [=] (int i) {
      if (array[i] != (1.0f * i)) {
        std::cout << "ERROR!" << std::endl;
      }
  });

  array.free();
}
