//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
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
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

#include <iostream>

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{

  std::cout << "Creating new arrays..." << std::endl;
  /*
   * Allocate two arrays with 10 elements of type float.
   */
  chai::ManagedArray<float> v1(10);
  chai::ManagedArray<float> v2(10);

  /*
   * Allocate an array on the device only
   */
  chai::ManagedArray<int> device_array(10, chai::GPU);

  std::cout << "Arrays created." << std::endl;

  std::cout << "Setting v1 on host." << std::endl;
  forall(sequential(), 0, 10, [=](int i) {
    v1[i] = static_cast<float>(i * 1.0f);
  });

  std::cout << "v1 = [";
  forall(sequential(), 0, 10, [=](int i) { std::cout << " " << v1[i]; });
  std::cout << " ]" << std::endl;

  std::cout << "Setting v2 and device_array on device." << std::endl;
  forall(gpu(), 0, 10, [=] __device__(int i) {
    v2[i] = v1[i] * 2.0f;
    device_array[i] = i;
  });

  std::cout << "v2 = [";
  forall(sequential(), 0, 10, [=](int i) { std::cout << " " << v2[i]; });
  std::cout << " ]" << std::endl;

  std::cout << "Doubling v2 on device." << std::endl;
  forall(gpu(), 0, 10, [=] __device__(int i) { v2[i] *= 2.0f; });

  std::cout << "Casting v2 to a pointer." << std::endl;
#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
  float* raw_v2 = v2;
#else
  float* raw_v2 = v2.data();
#endif

  std::cout << "raw_v2 = [";
  for (int i = 0; i < 10; i++) {
    std::cout << " " << raw_v2[i];
  }
  std::cout << " ]" << std::endl;

  std::cout << "Casting device_array to a pointer." << std::endl;
#if defined(CHAI_ENABLE_IMPLICIT_CONVERSIONS)
  int* raw_device_array = device_array;
#else  
  int* raw_device_array = device_array.data();
#endif
  std::cout << "device_array = [";
  for (int i = 0; i < 10; i++) {
    std::cout << " " << device_array[i];
  }
  std::cout << " ]" << std::endl;

  v1.free();
  v2.free();
  device_array.free();
}
