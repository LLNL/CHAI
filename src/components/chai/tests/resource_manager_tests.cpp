#include "gtest/gtest.h"

#include "chai/ResourceManager.hpp"

TEST(ResourceManager, Constructor) {
  chai::ResourceManager* rm = chai::ResourceManager::getResourceManager();

  ASSERT_NEQ(rm, nullptr);
}
