//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_HPP
#define CHAI_CONTEXT_HPP

namespace chai::expt
{
  enum class Context
  {
    NONE = 0,
    HOST = 1,
    DEVICE = 2
  };
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_HPP
