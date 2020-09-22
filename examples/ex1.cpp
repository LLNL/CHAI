//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/ManagedArray.hpp"
#include "chai/util/forall.hpp"

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{
  /*
   * Allocate an array.
   */
  chai::ManagedArray<double> array(50);

  /*
   * Fill data on the device
   */
  forall(gpu(), 0, 50, [=] __device__(int i) { array[i] = i * 2.0f; });

  /*
   * Print the array on the host, data is automatically copied back.
   */
  std::cout << "array = [";
  forall(sequential(), 0, 10, [=](int i) { std::cout << " " << array[i]; });
  std::cout << " ]" << std::endl;
}
