//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

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
    ContextRAJAPluginTester(const ContextRAJAPluginTester&)
      : m_context{::chai::expt::ContextManager::getInstance().getContext()}
    {
    }

    /*!
     * @brief Get the stored context.
     *
     * @return The stored ::chai::expt::Context value.
     */
    ::chai::expt::Context getContext() const {
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
TEST(ContextRAJAPlugin, ContextRAJAPlugin) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
    EXPECT_EQ(tester.getContext(), ::chai::expt::Context::HOST);
    EXPECT_EQ(::chai::expt::ContextManager::getInstance().getContext(), ::chai::expt::Context::NONE);
  });

  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
}
