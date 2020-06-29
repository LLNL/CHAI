//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "camp/resource.hpp"

#include "../src/util/forall.hpp"
#include "resource-example-util.hpp"

#include <vector>
#include <utility>


int main()
{
  constexpr std::size_t ARRAY_SIZE{1000};
  int clockrate{get_clockrate()};

  camp::resources::Resource dev1{camp::resources::Cuda{}};
  camp::resources::Resource dev2{camp::resources::Cuda{}};
  camp::resources::Resource host{camp::resources::Host{}};

  float * d_array1 = dev1.allocate<float>(1000);
  float * d_array2 = dev2.allocate<float>(1000);
  float * h_array1 = host.allocate<float>(1000);

  auto e1 = forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      d_array1[i] = i;
      gpu_time_wait_for(10, clockrate);
  });

  auto e2 = forall(&dev2, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      d_array2[i] = -1;
      gpu_time_wait_for(20, clockrate);
  });

  dev1.wait_for(&e2);

  forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      d_array1[i] *= d_array2[i];
      gpu_time_wait_for(10, clockrate);
  });

  dev1.memcpy(h_array1, d_array1, sizeof(float) * 1000);

  forall(&host, 0, 10, [=] CHAI_HOST_DEVICE (int i) {
      printf("%f ", h_array1[i]);
  });
  printf("\n");
}
