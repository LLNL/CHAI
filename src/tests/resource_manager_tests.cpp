#include "gtest/gtest.h"

#include "chai/ArrayManager.hpp"

TEST(ArrayManager, Constructor) {
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  ASSERT_NEQ(rm, nullptr);
}
