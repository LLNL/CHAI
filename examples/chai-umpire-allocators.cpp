//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"

#include "chai/ManagedArray.hpp"
#include "../src/util/forall.hpp"

#include <iostream>

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{

  auto& rm = umpire::ResourceManager::getInstance();

  auto cpu_pool =
      rm.makeAllocator<umpire::strategy::QuickPool>("cpu_pool",
                                                      rm.getAllocator("HOST"));

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  auto gpu_pool =
      rm.makeAllocator<umpire::strategy::QuickPool>("gpu_pool",
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
