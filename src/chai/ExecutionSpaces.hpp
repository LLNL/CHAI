//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ExecutionSpaces_HPP
#define CHAI_ExecutionSpaces_HPP

#include "chai/config.hpp"
#include "camp/resource.hpp"

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
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  /*! Execution in GPU space */
  GPU,
#endif
#if defined(CHAI_ENABLE_UM)
  UM,
#endif
#if defined(CHAI_ENABLE_PINNED)
  PINNED,
#endif
  // NUM_EXECUTION_SPACES should always be last!
  /*! Used to count total number of spaces */
  NUM_EXECUTION_SPACES
#if !defined(CHAI_ENABLE_CUDA) && !defined(CHAI_ENABLE_HIP) && !defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  ,GPU
#endif
#if !defined(CHAI_ENABLE_UM)
  ,UM
#endif
#if !defined(CHAI_ENABLE_PINNED)
  ,PINNED
#endif
};
static std::vector<char*> PrintExecSpace = { 
  (char *)"NONE",
  (char *)"CPU",
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  (char *)"GPU",
#endif
#if defined(CHAI_ENABLE_UM)
  (char *)"UM",
#endif
#if defined(CHAI_ENABLE_PINNED)
  (char *)"PINNED",
#endif
  (char *)"NUM_EXECUTION_SPACES"
};


inline bool operator==(const ExecutionSpace& s, const camp::resources::Platform& p) {
  if(s == chai::CPU && p == camp::resources::Platform::host) return true;
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  /*! Execution in GPU space */
  if (s == chai::GPU && (p == camp::resources::Platform::cuda ||
	                 p == camp::resources::Platform::hip)) return true;
#endif
  return false;
}

}  // end of namespace chai

#endif  // CHAI_ExecutionSpaces_HPP
