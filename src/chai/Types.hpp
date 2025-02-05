//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_Types_HPP
#define CHAI_Types_HPP

// Std library headers
#include <functional>

// CHAI headers
#include "chai/ExecutionSpaces.hpp"

#if defined(_WIN32) && !defined(CHAISTATICLIB)
#ifdef CHAISHAREDDLL_EXPORTS
#define CHAISHAREDDLL_API __declspec(dllexport)
#else
#define CHAISHAREDDLL_API __declspec(dllimport)
#endif
#else
#define CHAISHAREDDLL_API
#endif

namespace chai
{
  struct PointerRecord;

  typedef unsigned int uint;

  enum Action { ACTION_ALLOC, ACTION_FREE, ACTION_MOVE, ACTION_CAPTURED, ACTION_FOUND_ABANDONED, ACTION_LEAKED };

  using UserCallback = std::function<void(const PointerRecord*, Action, ExecutionSpace)>;
} // end of namespace chai


#endif
