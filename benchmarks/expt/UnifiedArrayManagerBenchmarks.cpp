//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include "chai/config.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/UnifiedArrayManager.hpp"
#include "camp/helpers.hpp"

#include <cstddef>
#include <cstdint>

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(CHAI_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {
  using ::chai::expt::Context;
  using ::chai::expt::ContextGuard;
  using ::chai::expt::ContextManager;

  inline void device_synchronize()
  {
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
#endif
  }

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  __global__ void touch_kernel(const std::int32_t* data, std::size_t size, std::int32_t* out)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      atomicAdd(out, data[i]);
    }
  }

  __global__ void add_constant_kernel(std::int32_t* data, std::size_t size, std::int32_t value)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      data[i] += value;
    }
  }

  __global__ void write_one_kernel(std::int32_t* data)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      data[0] += 1;
    }
  }

  __global__ void copy_kernel(const std::int32_t* in, std::int32_t* out, std::size_t size)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      out[i] = in[i];
    }
  }

  inline void launch_touch(const std::int32_t* data, std::size_t size, std::int32_t* out)
  {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

#if defined(CHAI_ENABLE_CUDA)
    touch_kernel<<<grid_size, BLOCK_SIZE>>>(data, size, out);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(touch_kernel,
                       dim3(grid_size),
                       dim3(BLOCK_SIZE),
                       0,
                       0,
                       data,
                       size,
                       out);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }

  inline void launch_add_constant(std::int32_t* data, std::size_t size, std::int32_t value)
  {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

#if defined(CHAI_ENABLE_CUDA)
    add_constant_kernel<<<grid_size, BLOCK_SIZE>>>(data, size, value);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(add_constant_kernel,
                       dim3(grid_size),
                       dim3(BLOCK_SIZE),
                       0,
                       0,
                       data,
                       size,
                       value);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }

  inline void launch_write_one(std::int32_t* data)
  {
#if defined(CHAI_ENABLE_CUDA)
    write_one_kernel<<<1, 1>>>(data);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(write_one_kernel, dim3(1), dim3(1), 0, 0, data);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }

  inline void launch_copy(const std::int32_t* in, std::int32_t* out, std::size_t size)
  {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

#if defined(CHAI_ENABLE_CUDA)
    copy_kernel<<<grid_size, BLOCK_SIZE>>>(in, out, size);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(copy_kernel,
                       dim3(grid_size),
                       dim3(BLOCK_SIZE),
                       0,
                       0,
                       in,
                       out,
                       size);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }
#endif

  inline void reset_context()
  {
    ContextManager::getInstance().reset();
  }

  inline void host_fill(::chai::expt::UnifiedArrayManager<std::int32_t>& manager, std::size_t size)
  {
    ContextGuard guard{Context::HOST};
    std::int32_t* data = manager.data(true);
    for (std::size_t i = 0; i < size; ++i)
    {
      data[i] = static_cast<std::int32_t>(i);
    }
    benchmark::ClobberMemory();
  }

  static void UnifiedArrayManagerHostWrite(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};

    for (auto _ : state)
    {
      host_fill(manager, size);
      benchmark::DoNotOptimize(manager.data(false));
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
  }

  static void UnifiedArrayManagerResize(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{};

    for (auto _ : state)
    {
      manager.resize(size);
      benchmark::DoNotOptimize(manager.data(false));
      manager.resize(0);
      benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
  }

  static void UnifiedArrayManagerHostDataAccess(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    ContextGuard guard{Context::HOST};
    for (auto _ : state)
    {
      benchmark::DoNotOptimize(manager.data(false));
    }
  }

  static void UnifiedArrayManagerDeviceDataAccess(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    ContextGuard guard{Context::DEVICE};
    for (auto _ : state)
    {
      benchmark::DoNotOptimize(manager.data(false));
    }

    device_synchronize();
  }

  static void UnifiedArrayManagerDeviceReadAfterHostWrite(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    // Managed output to prevent host reads from forcing extra copies.
    std::int32_t* out = nullptr;
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&out, sizeof(std::int32_t));
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&out, sizeof(std::int32_t));
#endif
    *out = 0;

    for (auto _ : state)
    {
      {
        ContextGuard guard{Context::DEVICE};
        const std::int32_t* data = manager.data(false);
        launch_touch(data, size, out);
      }
      device_synchronize();
      benchmark::DoNotOptimize(*out);
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));

#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*)out);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*)out);
#endif
  }

  static void UnifiedArrayManagerDeviceCopyAfterHostWrite(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    std::int32_t* out = nullptr;
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&out, sizeof(std::int32_t) * size);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&out, sizeof(std::int32_t) * size);
#endif

    for (auto _ : state)
    {
      {
        ContextGuard guard{Context::DEVICE};
        const std::int32_t* data = manager.data(false);
        launch_copy(data, out, size);
      }
      device_synchronize();
      benchmark::DoNotOptimize(out[0]);
      benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));

#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*)out);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*)out);
#endif
  }

  static void UnifiedArrayManagerHostReadAfterDeviceWriteAlreadySynced(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    for (auto _ : state)
    {
      state.PauseTiming();
      {
        ContextGuard guard{Context::DEVICE};
        std::int32_t* data = manager.data(true);
        launch_write_one(data);
      }
      device_synchronize();
      state.ResumeTiming();

      {
        ContextGuard guard{Context::HOST};
        std::int32_t* data = manager.data(false);
        benchmark::DoNotOptimize(data[0]);
      }
    }
  }

  static void UnifiedArrayManagerHostReadAfterDeviceWriteImplicitSync(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    for (auto _ : state)
    {
      {
        ContextGuard guard{Context::DEVICE};
        std::int32_t* data = manager.data(true);
        launch_write_one(data);
      }

      {
        ContextGuard guard{Context::HOST};
        std::int32_t* data = manager.data(false);
        benchmark::DoNotOptimize(data[0]);
      }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
  }

  static void UnifiedArrayManagerDeviceWriteThenHostReadSync(benchmark::State& state)
  {
    reset_context();
    const std::size_t size = static_cast<std::size_t>(state.range(0));

    ::chai::expt::UnifiedArrayManager<std::int32_t> manager{size};
    host_fill(manager, size);

    for (auto _ : state)
    {
      {
        ContextGuard guard{Context::DEVICE};
        std::int32_t* data = manager.data(true);
        launch_add_constant(data, size, 1);
      }

      // Host access should trigger synchronization since the array was touched
      // in the DEVICE context.
      {
        ContextGuard guard{Context::HOST};
        std::int32_t* data = manager.data(false);
        benchmark::DoNotOptimize(data[0]);
      }
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
  }
}  // namespace

BENCHMARK(UnifiedArrayManagerResize)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerHostDataAccess)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerDeviceDataAccess)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerHostWrite)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerDeviceReadAfterHostWrite)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerDeviceCopyAfterHostWrite)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerHostReadAfterDeviceWriteAlreadySynced)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerHostReadAfterDeviceWriteImplicitSync)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);
BENCHMARK(UnifiedArrayManagerDeviceWriteThenHostReadSync)->RangeMultiplier(2)->Range(1 << 10, 1 << 22);

BENCHMARK_MAIN();
