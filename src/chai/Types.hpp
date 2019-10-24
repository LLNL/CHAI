//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_Types_HPP
#define CHAI_Types_HPP

#include <functional>

namespace chai
{

typedef unsigned int uint;

enum Action { ACTION_ALLOC, ACTION_FREE, ACTION_MOVE };

using UserCallback = std::function<void(Action, ExecutionSpace, size_t)>;

} // end of namespace chai


#endif
