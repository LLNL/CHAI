//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ChaiMacros_HPP
#define CHAI_ChaiMacros_HPP

#include "chai/config.hpp"

#include "umpire/util/Macros.hpp"

#if defined(CHAI_ENABLE_CUDA) && defined(__CUDACC__)

#define CHAI_HOST __host__
#define CHAI_DEVICE __device__
#define CHAI_HOST_DEVICE __device__ __host__

#elif defined(CHAI_ENABLE_HIP) && defined(__HIPCC__)

#include <hip/hip_runtime.h>

#define CHAI_HOST __host__
#define CHAI_DEVICE __device__
#define CHAI_HOST_DEVICE __device__ __host__

#else

#define CHAI_HOST
#define CHAI_DEVICE
#define CHAI_HOST_DEVICE

#endif

#define CHAI_INLINE inline

#define CHAI_UNUSED_ARG(X)

#if !defined(CHAI_DISABLE_RM)

#define CHAI_LOG(level, msg) \
  UMPIRE_LOG(level, msg);

#else

#if defined(CHAI_DEBUG)

#define CHAI_LOG(level, msg) \
  std::cerr << "[" << __FILE__ << "] " << msg << std::endl;

#else

#define CHAI_LOG(level, msg)

#endif
#endif

#endif  // CHAI_ChaiMacros_HPP
