//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "RAJA/RAJA.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/ManagedArrayView.hpp"

#include <iostream>
#include <chrono>

static constexpr int SIZE = 10000000;

int main(int CHAI_UNUSED_ARG(argc), char** CHAI_UNUSED_ARG(argv))
{
    chai::ManagedArray<float> array(SIZE);
#ifdef USE_MA_VIEW
    using view_t = chai::ManagedArrayView<float, RAJA::Layout<1> >;
    view_t view(array, SIZE);
#else
    using view_t = RAJA::View<float, RAJA::Layout<1> >;
    float* ptr = array;
    view_t view(ptr, SIZE);
#endif

    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, SIZE), [=](int i) {
      view(i) = static_cast<float>(i * 1.0f);
    });
}
