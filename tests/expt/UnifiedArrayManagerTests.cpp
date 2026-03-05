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
  __global__ void read_samples_kernel(const int* data,
                                      std::size_t size,
                                      int* out_samples /* length 3 */)
  {
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
      out_samples[0] = data[0];
      out_samples[1] = data[size / 2];
      out_samples[2] = data[size - 1];
    }
  }

  __global__ void increment_kernel(int* data, std::size_t size)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      data[i] += 1;
    }
  }

  inline void launch_read_samples(const int* data, std::size_t size, int* out_samples)
  {
#if defined(CHAI_ENABLE_CUDA)
    read_samples_kernel<<<1, 1>>>(data, size, out_samples);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(read_samples_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       data,
                       size,
                       out_samples);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
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

TEST_F(UnifiedArrayManagerTest, HostReadWrite)
{
  constexpr std::size_t N = 64;
  ::chai::expt::UnifiedArrayManager<int> manager{N};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
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

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i));
    }
  }
}

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

TEST_F(UnifiedArrayManagerTest, DeviceReadDoesNotSynchronizeOnHostAccess)
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

  int* samples = malloc_managed<int>(3);
  ASSERT_NE(samples, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_read_samples(data, N, samples);
  }

  // The DEVICE context has been entered, but we haven't synchronized yet.
  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  // Since the array was not "touched" in the DEVICE context, host access should
  // not trigger synchronization.
  {
    ContextGuard guard{Context::HOST};
    (void)manager.data(false);
    EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  }

  // We still need to synchronize before reading the kernel output.
  contextManager.synchronize(Context::DEVICE);
  EXPECT_TRUE(contextManager.isSynchronized(Context::DEVICE));
  EXPECT_EQ(samples[0], 0);
  EXPECT_EQ(samples[1], static_cast<int>(N / 2));
  EXPECT_EQ(samples[2], static_cast<int>(N - 1));

  free_managed(samples);
}

TEST_F(UnifiedArrayManagerTest, HostToDeviceAccessDoesNotSynchronizeDevice)
{
  constexpr std::size_t N = 64;
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
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  {
    ContextGuard guard{Context::HOST};
    (void)manager.data(false);
    EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  }
}

TEST_F(UnifiedArrayManagerTest, HostReadThenDeviceRead)
{
  constexpr std::size_t N = 256;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  // Host read without touching does not mark the data as modified in HOST.
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data[0], 0);
  }

  int* samples = malloc_managed<int>(3);
  ASSERT_NE(samples, nullptr);

  // Device read without touching does not require device synchronization, and does not
  // cause later host access to synchronize the device.
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_read_samples(data, N, samples);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  {
    ContextGuard guard{Context::HOST};
    (void)manager.data(false);
    EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  }

  contextManager.synchronize(Context::DEVICE);
  EXPECT_EQ(samples[0], 0);
  EXPECT_EQ(samples[1], 0);
  EXPECT_EQ(samples[2], 0);

  free_managed(samples);
}

TEST_F(UnifiedArrayManagerTest, HostReadThenDeviceWriteThenHostReadSynchronizes)
{
  constexpr std::size_t N = 128;
  ::chai::expt::UnifiedArrayManager<int> manager{N};
  ContextManager& contextManager = ContextManager::getInstance();

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data[0], 0);
  }

  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    EXPECT_TRUE(contextManager.isSynchronized(Context::DEVICE));
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[N - 1], 1);
  }
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenHostWriteDoesNotSynchronize)
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

  int* samples = malloc_managed<int>(3);
  ASSERT_NE(samples, nullptr);

  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_read_samples(data, N, samples);
  }

  EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));

  // Since the array was not touched in DEVICE, host write should not synchronize DEVICE.
  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    data[0] = -7;
    EXPECT_FALSE(contextManager.isSynchronized(Context::DEVICE));
  }

  contextManager.synchronize(Context::DEVICE);
  EXPECT_EQ(samples[0], 0);
  EXPECT_EQ(samples[1], static_cast<int>(N / 2));
  EXPECT_EQ(samples[2], static_cast<int>(N - 1));

  free_managed(samples);
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenHostWriteSynchronizes)
{
  constexpr std::size_t N = 128;
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

TEST_F(UnifiedArrayManagerTest, DeviceWriteSynchronizesOnHostAccess)
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

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    // Since the array was "touched" in the DEVICE context, host access triggers
    // synchronization before returning the pointer.
    EXPECT_TRUE(contextManager.isSynchronized(Context::DEVICE));

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i + 1));
    }
  }
}
