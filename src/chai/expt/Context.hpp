//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_HPP
#define CHAI_CONTEXT_HPP

namespace chai {
namespace expt {
  /*!
   * \enum Context
   *
   * \brief Represents the state of a program. ArrayManagers update coherence based on the context.
   */
  enum Context {
    NONE = 0,  ///< Represents no context.
    HOST       ///< Represents the host context (i.e. the CPU).
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
    , DEVICE   ///< Represents the device context (i.e. the GPU).
#endif
  };
}  // namespace expt
}  // namespace chai

#endif  // CHAI_CONTEXT_HPP