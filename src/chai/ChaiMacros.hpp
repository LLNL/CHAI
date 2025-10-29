//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ChaiMacros_HPP
#define CHAI_ChaiMacros_HPP

#include "chai/config.hpp"

#include "umpire/util/Macros.hpp"

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
#define CHAI_ENABLE_DEVICE
#endif

#if defined(CHAI_ENABLE_CUDA)

#include <cuda_runtime_api.h>

#define CHAI_HOST __host__
#define CHAI_DEVICE __device__
#define CHAI_HOST_DEVICE __device__ __host__
#define CHAI_GLOBAL __global__

#define gpuMemcpyKind cudaMemcpyKind
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault cudaMemcpyDefault

#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuGetErrorString cudaGetErrorString
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuDeviceSynchronize cudaDeviceSynchronize


// NOTE: Cannot have if defined(__HIPCC__) in the condition below, since __HIPCC__ comes from the included header hip_runtime below.
#elif defined(CHAI_ENABLE_HIP)

#include <hip/hip_runtime.h>

#define CHAI_HOST __host__
#define CHAI_DEVICE __device__
#define CHAI_HOST_DEVICE __device__ __host__
#define CHAI_GLOBAL __global__

#define gpuMemcpyKind hipMemcpyKind
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDefault hipMemcpyDefault

#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuGetErrorString hipGetErrorString
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuDeviceSynchronize hipDeviceSynchronize

#else

#define CHAI_HOST
#define CHAI_DEVICE
#define CHAI_HOST_DEVICE
#define CHAI_GLOBAL

#define gpuMemcpyKind int
#define gpuMemcpyHostToHost 0
#define gpuMemcpyHostToDevice 1
#define gpuMemcpyDeviceToHost 2
#define gpuMemcpyDeviceToDevice 3
#define gpuMemcpyDefault 4

#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define CHAI_DEVICE_COMPILE
#endif

// shorthand for GPU Compilation. Must go after hip/hip_runtime.h is included so that HIPCC is defined
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHAI_GPUCC
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

namespace chai
{
template <typename... T>
CHAI_HOST_DEVICE CHAI_INLINE void CHAI_UNUSED_VAR(T &&...) noexcept {}
} // namespace chai

#endif  // CHAI_ChaiMacros_HPP
