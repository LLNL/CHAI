//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
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

  chai::ManagedArray<float> array_one(4096, chai::PINNED);
  chai::ManagedArray<float> array_two(4096, chai::PINNED);
  chai::ManagedArray<float> array_three(4096, chai::PINNED);
  chai::ManagedArray<float> array_four(4096, chai::PINNED);
  chai::ManagedArray<float> array_five(4096, chai::PINNED);

  std::cout << "Setting arrays on host." << std::endl;
  forall(sequential(), 0, 4096, [=](int i) {
    array_one[i] = static_cast<float>(i * 1.0f);
    array_two[i] = static_cast<float>(i * 2.0f);
    array_three[i] = static_cast<float>(i * 3.0f);
    array_four[i] = static_cast<float>(i * 4.0f);
    array_five[i] = static_cast<float>(i * 5.0f);
  });
  
  forall(sequential(), 0, 3, [=](int i) {
    std::cout << array_one[i] << " "; 
    std::cout << array_two[i] << " ";
    std::cout << array_three[i] << " ";
    std::cout << array_four[i] << " ";
    std::cout << array_five[i] << " ";
  });
  std::cout << std::endl;

  std::cout << "Doubling on device." << std::endl;
  forall(gpu_async(), 0, 4096, [=] __device__(int i) {
    array_one[i] *= 2.0f;
    array_two[i] *= 2.0f;
    array_three[i] *= 2.0f;
    array_four[i] *= 2.0f;
    array_five[i] *= 2.0f;
  });

  forall(sequential(), 0, 3, [=](int i) {
    std::cout << array_one[i] << " "; 
    std::cout << array_two[i] << " ";
    std::cout << array_three[i] << " ";
    std::cout << array_four[i] << " ";
    std::cout << array_five[i] << " ";
  });
  std::cout << std::endl;

  array_one.free();
  array_two.free();
  array_three.free();
  array_four.free();
  array_five.free();

 return 0;
}
