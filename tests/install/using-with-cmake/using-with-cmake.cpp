//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and CHAI project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "chai/ManagedArray.hpp"
#include <cstddef>

int main(int, char**) 
{
  constexpr std::size_t N{1024};
  chai::ManagedArray<std::size_t> a(N);

  for (std::size_t i = 0; i < N; i++) {
    a[i] = i;
  }

  a.free();
}
