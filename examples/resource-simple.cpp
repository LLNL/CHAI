//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "../src/util/forall.hpp"
#include "../src/util/gpu_clock.hpp"

#include "chai/ManagedArray.hpp"
#include "camp/resource.hpp"

#include <vector>
#include <utility>


int main()
{

  constexpr int NUM_ARRAYS = 16;
  constexpr std::size_t ARRAY_SIZE{100};

  std::vector<chai::ManagedArray<double>> arrays;
  camp::resources::Resource host{camp::resources::Host{}}; 


  int clockrate{get_clockrate()};

  for (std::size_t i = 0; i < NUM_ARRAYS; ++i) {
    arrays.push_back(chai::ManagedArray<double>(ARRAY_SIZE));
  }

  for (auto array : arrays) {
    forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
        array[i] = i;
    }); 
  }

  for (auto array : arrays) {
    camp::resources::Resource resource{camp::resources::Cuda{}}; 

    forall(&resource, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
        array[i] = array[i] * 2.0;
        gpu_time_wait_for(20, clockrate);
    });

    array.move(chai::CPU, &resource);
  }

  for (auto array : arrays) {
    forall(&host, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
        if (i == 25) {
          printf("array[%d] = %f \n", i, array[i]);
        }
    }); 
  }
}
