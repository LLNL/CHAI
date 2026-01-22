//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/HostArrayManager.hpp"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdlib>

TEST(HostArrayManager, DefaultConstructor) {
  ::chai::expt::HostArrayManager<int> a;
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::NONE};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::DEVICE};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
  }
}

TEST(HostArrayManager, AllocatorConstructor) {
  ::chai::expt::HostArrayManager<int> a{umpire::ResourceManager::getInstance().getAllocator("HOST")};
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.data(), nullptr);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::NONE};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::DEVICE};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);
  }
}

TEST(HostArrayManager, SizeConstructor) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size};
  EXPECT_EQ(a.size(), size);
  EXPECT_EQ(a.data(), nullptr);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::NONE};
    EXPECT_EQ(a.size(), size);
    EXPECT_EQ(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), size);
    EXPECT_NE(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::DEVICE};
    EXPECT_EQ(a.size(), size);
    EXPECT_EQ(a.data(), nullptr);
  }
}

TEST(HostArrayManager, SizeAndAllocatorConstructor) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size, umpire::ResourceManager::getInstance().getAllocator("HOST")};
  EXPECT_EQ(a.size(), size);
  EXPECT_EQ(a.data(), nullptr);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::NONE};
    EXPECT_EQ(a.size(), size);
    EXPECT_EQ(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), size);
    EXPECT_NE(a.data(), nullptr);
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::DEVICE};
    EXPECT_EQ(a.size(), size);
    EXPECT_EQ(a.data(), nullptr);
  }
}

TEST(HostArrayManager, CopyConstructor) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size, umpire::ResourceManager::getInstance().getAllocator("HOST")};

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    int* data = a.data();

    for (std::size_t i = 0; i < size; ++i)
    {
      data[i] = i;
    }
  }

  ::chai::expt::HostArrayManager<int> b{a};

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), b.size());

    const int* a_data = a.data();
    const int* b_data = b.data();

    EXPECT_NE(a_data, b_data);

    for (std::size_t i = 0; i < size; ++i)
    {
      EXPECT_EQ(a_data[i], i);
      EXPECT_EQ(b_data[i], i);
    }
  }
}

TEST(HostArrayManager, CopyAssignmentOperator) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size, umpire::ResourceManager::getInstance().getAllocator("HOST")};

  int* data = nullptr;

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    data = a.data();

    for (std::size_t i = 0; i < size; ++i)
    {
      data[i] = i;
    }
  }

  ::chai::expt::HostArrayManager<int> b;
  b = a;

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), size);
    const int* a_data = a.data();
    EXPECT_EQ(a_data, data);

    for (std::size_t i = 0; i < size; ++i)
    {
      EXPECT_EQ(a_data[i], i);
    }

    EXPECT_EQ(b.size(), size);
    const int* b_data = b.data();
    EXPECT_NE(b_data, nullptr);
    EXPECT_NE(b_data, a_data);

    for (std::size_t i = 0; i < size; ++i)
    {
      EXPECT_EQ(b_data[i], i);
    }
  }
}

TEST(HostArrayManager, MoveConstructor) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size, umpire::ResourceManager::getInstance().getAllocator("HOST")};

  int* data = nullptr;

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    data = a.data();

    for (std::size_t i = 0; i < size; ++i)
    {
      data[i] = i;
    }
  }

  ::chai::expt::HostArrayManager<int> b{std::move(a)};

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);

    EXPECT_EQ(b.size(), size);
    const int* b_data = b.data();
    EXPECT_EQ(b_data, data);

    for (std::size_t i = 0; i < size; ++i)
    {
      EXPECT_EQ(b_data[i], i);
    }
  }
}

TEST(HostArrayManager, MoveAssignmentOperator) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size, umpire::ResourceManager::getInstance().getAllocator("HOST")};

  int* data = nullptr;

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    data = a.data();

    for (std::size_t i = 0; i < size; ++i)
    {
      data[i] = i;
    }
  }

  ::chai::expt::HostArrayManager<int> b;
  b = std::move(a);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.data(), nullptr);

    EXPECT_EQ(b.size(), size);
    const int* b_data = b.data();
    EXPECT_EQ(b_data, data);

    for (std::size_t i = 0; i < size; ++i)
    {
      EXPECT_EQ(b_data[i], i);
    }
  }
}

TEST(HostArrayManager, Resize) {
  const std::size_t size = 10;
  ::chai::expt::HostArrayManager<int> a{size, umpire::ResourceManager::getInstance().getAllocator("HOST")};

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    int* data = a.data();

    for (std::size_t i = 0; i < size; ++i)
    {
      data[i] = i;
    }
  }

  const int new_size = 5;
  a.resize(new_size);
  EXPECT_EQ(a.size(), new_size);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    int* data = a.data();

    for (std::size_t i = 0; i < new_size; ++i)
    {
      EXPECT_EQ(data[i], i);
    }
  }

  a.resize(size);
  EXPECT_EQ(a.size(), size);

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    int* data = a.data();

    for (std::size_t i = 0; i < new_size; ++i)
    {
      EXPECT_EQ(data[i], i);
    }
  }
}
