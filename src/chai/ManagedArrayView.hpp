//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ManagedArrayView_HPP
#define CHAI_ManagedArrayView_HPP

#include "chai/config.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)

#include "chai/ManagedArray.hpp"

#include "RAJA/util/View.hpp"

namespace chai {

  template <typename ValueType, typename LayoutType>
using ManagedArrayView =
    RAJA::View<ValueType, LayoutType, chai::ManagedArray<ValueType>>;


template <typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedManagedArrayView = RAJA::TypedViewBase<ValueType,
                                            chai::ManagedArray<ValueType>,
                                            LayoutType,
                                            IndexTypes...>;

} // end of namespace chai

#endif // defined(CHAI_ENABLE_RAJA_PLUGIN)

#endif // CHAI_ManagedArrayView_HPP
