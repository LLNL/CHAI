//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

//
// A c-style stack array containing chai::ManagedArray<int> crashes
// at run time with the following error:
//
// corrupted double-linked list
// flux-job: task(s) exited with exit code 134
//
// During the process of tweaking this reproducer the following errors
// have also occurred:
//
// malloc_consolidate(): unaligned fastbin chunk detected
// flux-job: task(s) exited with exit code 134
//
// flux-job: task(s) exited with exit code 139
//

#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"

int main(int, char**) {
   // Kernel using c-style stack array of chai::ManagedArrays
   chai::ManagedArray<int> a[1];

   for (int i = 0; i < 1; ++i) {
      a[i] = chai::ManagedArray<int>(10);
   }

   RAJA::forall<RAJA::hip_exec<256, true>>(
      RAJA::RangeSegment(0, 10),
      [=] __device__ (int i) { a[0][i] = i; });

   a[0].free();

   // Kernel afterwards
   int b[2] = {3, 7};

   RAJA::forall<RAJA::hip_exec<256, true>>(
      RAJA::RangeSegment(0, 2),
      [=] __device__ (int i) { static_cast<void>(b[i]); });

   return 0;
}

