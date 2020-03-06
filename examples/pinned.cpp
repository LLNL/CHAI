//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

#include <iostream>

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{
  std::cout << "Creating a pinned array..." << std::endl;

  chai::ManagedArray<float> array(10, chai::PINNED);

  std::cout << "Setting array on host." << std::endl;
  std::cout << "array = [";
  forall(sequential(), 0, 10, [=](int i) {
    array[i] = static_cast<float>(i * 1.0f);
    std::cout << " " << array[i];
  });
  std::cout << " ]" << std::endl;


  std::cout << "Doubling on device." << std::endl;
  forall(gpu_async(), 0, 10, [=] __device__(int i) { array[i] *= 2.0f; });

  std::cout << "array = [";
  forall(sequential(), 0, 10, [=](int i) {
    std::cout << " " << array[i];
  });
  std::cout << " ]" << std::endl;

  array.free();
}
