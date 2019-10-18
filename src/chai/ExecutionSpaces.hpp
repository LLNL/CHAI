//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ExecutionSpaces_HPP
#define CHAI_ExecutionSpaces_HPP

#include "chai/config.hpp"
#include "camp/device.hpp"

namespace chai
{

/*!
 * \brief Enum listing possible execution spaces.
 */
enum ExecutionSpace {
  /*! Default, no execution space. */
  NONE = 0,
  /*! Executing in CPU space */
  CPU,
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  /*! Execution in GPU space */
  GPU,
#endif
#if defined(CHAI_ENABLE_UM)
  UM,
#endif
  // NUM_EXECUTION_SPACES should always be last!
  /*! Used to count total number of spaces */
  NUM_EXECUTION_SPACES
};

inline bool operator==(const ExecutionSpace& s, const camp::devices::Platform& p) {
  if(s == chai::CPU && p == camp::devices::Platform::host) return true;
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  /*! Execution in GPU space */
  if (s == chai::GPU && (p == camp::devices::Platform::cuda ||
	                 p == camp::devices::Platform::hip)) return true;
#endif
  return false;
}

}  // end of namespace chai

#endif  // CHAI_ExecutionSpaces_HPP
