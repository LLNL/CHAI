//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_HPP
#define CHAI_EXECUTION_CONTEXT_HPP

#include "chai/config.hpp"

#include <optional>
#include <variant>

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(CHAI_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace chai::expt
{
  enum class SyncPolicy
  {
    Synchronous,
    Asynchronous
  };

  struct HostContext
  {
    SyncPolicy policy{SyncPolicy::Synchronous};
  };

  inline bool operator==(const HostContext& a, const HostContext& b)
  {
    return a.policy == b.policy;
  }

#if defined(CHAI_ENABLE_CUDA)
  struct CudaContext
  {
    cudaStream_t stream{};
    SyncPolicy policy{SyncPolicy::Asynchronous};
  };

  inline bool operator==(const CudaContext& a, const CudaContext& b)
  {
    return a.stream == b.stream && a.policy == b.policy;
  }
#endif

#if defined(CHAI_ENABLE_HIP)
  struct HipContext
  {
    hipStream_t stream{};
    SyncPolicy policy{SyncPolicy::Asynchronous};
  };

  inline bool operator==(const HipContext& a, const HipContext& b)
  {
    return a.stream == b.stream && a.policy == b.policy;
  }
#endif

  using ExecutionContext =
    std::variant<
      HostContext
#if defined(CHAI_ENABLE_CUDA)
      ,
      CudaContext
#endif
#if defined(CHAI_ENABLE_HIP)
      ,
      HipContext
#endif
      >;

  using OptionalExecutionContext = std::optional<ExecutionContext>;
}  // namespace chai::expt

#endif  // CHAI_EXECUTION_CONTEXT_HPP
