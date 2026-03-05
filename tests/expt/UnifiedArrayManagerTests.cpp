//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/UnifiedArrayManager.hpp"
#include "camp/helpers.hpp"
#include "gtest/gtest.h"

#include <cstddef>

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

  template <typename T>
  T* malloc_managed(std::size_t count)
  {
    T* ptr = nullptr;

#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&ptr, sizeof(T) * count);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&ptr, sizeof(T) * count);
#else
    static_cast<void>(count);
#endif

    return ptr;
  }

  inline void free_managed(void* ptr)
  {
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, ptr);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, ptr);
#else
    static_cast<void>(ptr);
#endif
  }

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  __global__ void increment_kernel(int* data, std::size_t size)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      data[i] += 1;
    }
  }

  __global__ void copy_kernel(const int* in, int* out, std::size_t size)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      out[i] = in[i];
    }
  }

  inline void device_synchronize_raw()
  {
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
#endif
  }

  inline void launch_increment(int* data, std::size_t size)
  {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

#if defined(CHAI_ENABLE_CUDA)
    increment_kernel<<<grid_size, BLOCK_SIZE>>>(data, size);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(increment_kernel,
                       dim3(grid_size),
                       dim3(BLOCK_SIZE),
                       0,
                       0,
                       data,
                       size);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }

  inline void launch_copy(const int* in, int* out, std::size_t size)
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

  class UnifiedArrayManagerTest : public ::testing::Test
  {
    protected:
      void SetUp() override
      {
        ContextManager::getInstance().reset();
      }

      void TearDown() override
      {
        ContextManager::getInstance().reset();
      }
  };
}  // namespace

TEST_F(UnifiedArrayManagerTest, DefaultConstructorAndResizeToZero)
{
  ::chai::expt::UnifiedArrayManager<int> manager{};
  EXPECT_EQ(manager.size(), 0);

  {
    ContextGuard guard{Context::HOST};
    EXPECT_EQ(manager.data(false), nullptr);
    EXPECT_EQ(manager.data(true), nullptr);
  }

  manager.resize(0);
  EXPECT_EQ(manager.size(), 0);

  {
    ContextGuard guard{Context::HOST};
    EXPECT_EQ(manager.data(false), nullptr);
  }
}

TEST_F(UnifiedArrayManagerTest, ResizeGrowsAndValueInitializesNewElements)
{
  constexpr std::size_t N0 = 8;
  constexpr std::size_t N1 = 16;

  ::chai::expt::UnifiedArrayManager<int> manager{N0};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N0; ++i)
    {
      data[i] = static_cast<int>(i + 10);
    }
  }

  manager.resize(N1);
  EXPECT_EQ(manager.size(), N1);

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N0; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i + 10));
    }

    for (std::size_t i = N0; i < N1; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }
}

TEST_F(UnifiedArrayManagerTest, ResizeShrinksPreservesPrefix)
{
  constexpr std::size_t N0 = 16;
  constexpr std::size_t N1 = 6;

  ::chai::expt::UnifiedArrayManager<int> manager{N0};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N0; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  manager.resize(N1);
  EXPECT_EQ(manager.size(), N1);

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N1; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, UsesProvidedAllocator)
{
  constexpr std::size_t N = 32;
  auto& rm = ::umpire::ResourceManager::getInstance();
  umpire::Allocator allocator = rm.getAllocator("UM");

  ::chai::expt::UnifiedArrayManager<int> manager{N, allocator};
  EXPECT_EQ(manager.size(), N);

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
      data[i] = static_cast<int>(i * 2);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i * 2));
    }
  }
}

// Read/write matrix tests. These are placed at the end of the file so that
// constructor/resize/allocator behavior is tested first.

TEST_F(UnifiedArrayManagerTest, HostReadThenHostRead)
{
  constexpr std::size_t N = 128;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }
}

TEST_F(UnifiedArrayManagerTest, HostReadThenHostWrite)
{
  constexpr std::size_t N = 128;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  int* host_ptr = nullptr;

  {
    ContextGuard guard{Context::HOST};
    host_ptr = manager.data(false);
    ASSERT_NE(host_ptr, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(host_ptr[i], 0);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(host_ptr[i], static_cast<int>(i));
  }
}

TEST_F(UnifiedArrayManagerTest, HostReadThenDeviceRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  contextManager.synchronize(Context::DEVICE);
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 0);
  }

  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, HostReadThenDeviceWrite)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  int* host_ptr = nullptr;

  {
    ContextGuard guard{Context::HOST};
    host_ptr = manager.data(false);
    ASSERT_NE(host_ptr, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(host_ptr[i], 0);
    }
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  contextManager.synchronize(Context::DEVICE);
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(host_ptr[i], 1);
  }
}

TEST_F(UnifiedArrayManagerTest, HostWriteThenHostRead)
{
  constexpr std::size_t N = 128;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, HostWriteThenHostWrite)
{
  constexpr std::size_t N = 128;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = -static_cast<int>(i);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], -static_cast<int>(i));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, HostWriteThenDeviceRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  // Since the array was not touched in DEVICE, a later host access should not
  // trigger synchronization.
  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  contextManager.synchronize(Context::DEVICE);
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], static_cast<int>(i));
  }

  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, HostWriteThenDeviceWrite)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  int* host_ptr = nullptr;

  {
    ContextGuard guard{Context::HOST};
    host_ptr = manager.data(true);
    ASSERT_NE(host_ptr, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      host_ptr[i] = static_cast<int>(i);
    }
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  contextManager.synchronize(Context::DEVICE);
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(host_ptr[i], static_cast<int>(i + 1));
  }
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenHostRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  // Ensure kernel completion without updating ContextManager's sync state.
  device_synchronize_raw();
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 0);
  }

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
    EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  }

  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenHostWrite)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  device_synchronize_raw();

  // Since the array was not touched in DEVICE, host write should not synchronize DEVICE.
  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    data[0] = -7;
    EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  }

  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], static_cast<int>(i));
  }

  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenDeviceRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  int* out0 = malloc_managed<int>(N);
  int* out1 = malloc_managed<int>(N);
  ASSERT_NE(out0, nullptr);
  ASSERT_NE(out1, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out0, N);
  }

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out1, N);
  }

  device_synchronize_raw();
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out0[i], 0);
    EXPECT_EQ(out1[i], 0);
  }

  free_managed(out0);
  free_managed(out1);
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenDeviceWrite)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  int* out_after = malloc_managed<int>(N);
  ASSERT_NE(out_after, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out_after, N);
  }

  device_synchronize_raw();
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out_after[i], 0);
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
    launch_copy(data, out_after, N);
  }

  device_synchronize_raw();
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out_after[i], 1);
  }

  free_managed(out_after);
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenHostRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  // Since the most recent modification was in DEVICE, host access synchronizes first.
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    EXPECT_TRUE(contextManager.isSynchronized(Context::DEVICE));
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i + 1));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenHostWrite)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = 0;
    }
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  // Host write should synchronize first because the most recent modification was in DEVICE.
  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    EXPECT_TRUE(contextManager.isSynchronized(Context::DEVICE));
    data[0] = 42;
  }

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data[0], 42);
  }
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenDeviceRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  device_synchronize_raw();
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 1);
  }

  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenDeviceWrite)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
    launch_copy(data, out, N);
  }

  device_synchronize_raw();
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 2);
  }

  free_managed(out);
}
