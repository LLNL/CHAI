//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include "TestHelpers.hpp"

// Pre-main registration of plugin with RAJA
static ::RAJA::util::PluginRegistry::add<::chai::expt::ContextRAJAPlugin> P(
  "CHAIContextPlugin",
  "Plugin that integrates CHAI context management with RAJA.");

/*!
 * \brief Tests whether the plugin was actually called.
 */
class ContextRAJAPluginTester {
  public:
    /*!
     * @brief Construct a tester with an initial context of NONE.
     */
    ContextRAJAPluginTester() = default;

    /*!
     * @brief Copy-construct and capture the current ContextManager context.
     */
    CHAI_HOST_DEVICE ContextRAJAPluginTester(const ContextRAJAPluginTester& other)
      : m_context{other.m_context}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      ::chai::expt::Context context = ::chai::expt::ContextManager::getInstance().getContext();

      if (context != ::chai::expt::Context::NONE) {
        m_context = context;
      }
#endif
    }

    /*!
     * @brief Get the stored context.
     *
     * @return The stored ::chai::expt::Context value.
     */
    CHAI_HOST_DEVICE ::chai::expt::Context getContext() const {
      return m_context;
    }

  private:
    /*!
     * @brief Stored context value.
     */
    ::chai::expt::Context m_context{::chai::expt::Context::NONE};
};

// Test that the tester object got the updated context and that the current context
// is NONE inside the loop.
TEST(ContextRAJAPlugin, HOST) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
    EXPECT_EQ(tester.getContext(), ::chai::expt::Context::HOST);
    EXPECT_EQ(::chai::expt::ContextManager::getInstance().getContext(), ::chai::expt::Context::NONE);
  });

  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
// Test that the tester object got the updated context.
CUDA_TEST(ContextRAJAPlugin, CUDA) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::chai::expt::Context* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&result, sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
// Test that the tester object got the updated context.
TEST(ContextRAJAPlugin, HIP) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::chai::expt::Context* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&result, sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
}
#endif
