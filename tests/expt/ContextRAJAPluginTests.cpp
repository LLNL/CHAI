//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/expt/ExecutionContext.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include "TestHelpers.hpp"

#include <type_traits>

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
      ::chai::expt::OptionalExecutionContext context = ::chai::expt::ContextManager::getInstance().getContext();

      if (context)
      {
        m_context = std::visit(
          [](const auto& ctx) -> int {
            using ContextType = std::decay_t<decltype(ctx)>;
            if constexpr (std::is_same_v<ContextType, ::chai::expt::HostContext>)
            {
              return 1;
            }
            else
            {
              return 2;
            }
          },
          *context);
      }
#endif
    }

    /*!
     * @brief Get the stored context.
     *
     * @return 0 for unset, 1 for host, 2 for device.
     */
    CHAI_HOST_DEVICE int getContext() const {
      return m_context;
    }

  private:
    /*!
     * @brief Stored context value.
     */
    int m_context{0};
};

// Test that the tester object got the updated context and that the current context
// is NONE inside the loop.
TEST(ContextRAJAPlugin, HOST) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), 0);

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
    EXPECT_EQ(tester.getContext(), 1);
    EXPECT_FALSE(::chai::expt::ContextManager::getInstance().hasContext());
  });

  EXPECT_EQ(tester.getContext(), 0);
}

#if defined(CHAI_ENABLE_CUDA)
// Test that the tester object got the updated context.
CUDA_TEST(ContextRAJAPlugin, CUDA) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), 0);

  int* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&result, sizeof(int));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(*result, 2);
  EXPECT_EQ(tester.getContext(), 0);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
// Test that the tester object got the updated context.
TEST(ContextRAJAPlugin, HIP) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), 0);

  int* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&result, sizeof(int));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(*result, 2);
  EXPECT_EQ(tester.getContext(), 0);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
}
#endif
