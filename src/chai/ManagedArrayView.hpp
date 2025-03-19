//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
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
using TypedManagedArrayView = RAJA::internal::TypedViewBase<ValueType,
                                            chai::ManagedArray<ValueType>,
                                            LayoutType,
                                            camp::list<IndexTypes...> >;

template <typename ValueType, typename LayoutType, RAJA::Index_type P2Pidx = 0>
using ManagedArrayMultiView =
    RAJA::MultiView<ValueType,
                    LayoutType,
                    P2Pidx,
                    chai::ManagedArray<ValueType> *,
                    chai::ManagedArray<camp::type::cv::rem<ValueType>> *>;

} // end of namespace chai

#endif // defined(CHAI_ENABLE_RAJA_PLUGIN)

#endif // CHAI_ManagedArrayView_HPP
