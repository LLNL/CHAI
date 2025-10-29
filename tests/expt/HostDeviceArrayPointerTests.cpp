#include "chai/config.hpp"
#include "chai/expt/ArrayPointer.hpp"
#include "chai/expt/HostDeviceArrayManager.hpp"
#include <gtest/gtest.h>

namespace {

template <typename T>
using HostDeviceArrayPointer =
  chai::expt::ArrayPointer<T, chai::expt::HostDeviceArrayManager>;

class HostDeviceArrayPointerTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(HostDeviceArrayPointerTest, DefaultConstructor) {
  HostDeviceArrayPointer<int> ptr;
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.data(), nullptr);
}

TEST_F(HostDeviceArrayPointerTest, NullptrConstructor) {
  HostDeviceArrayPointer<double> ptr(nullptr);
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.data(), nullptr);
}

TEST_F(HostDeviceArrayPointerTest, ManagerConstructor) {
  auto* manager = new chai::expt::HostDeviceArrayManager<int>(5);
  HostDeviceArrayPointer<int> ptr(manager);
  
  EXPECT_EQ(ptr.size(), 5);
  EXPECT_NE(ptr.data(), nullptr);
  
  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, CopyConstructor) {
  auto* manager = new chai::expt::HostDeviceArrayManager<int>(5);
  HostDeviceArrayPointer<int> ptr1(manager);
  HostDeviceArrayPointer<int> ptr2(ptr1);
  
  EXPECT_EQ(ptr2.size(), 5);
  EXPECT_NE(ptr2.data(), nullptr);
  
  ptr1.free();
}

TEST_F(HostDeviceArrayPointerTest, ConvertingConstructor) {
  auto* manager = new chai::expt::HostDeviceArrayManager<int>(5);
  HostDeviceArrayPointer<int> ptr1(manager);
  HostDeviceArrayPointer<const int> ptr2(ptr1);
  
  EXPECT_EQ(ptr2.size(), 5);
  EXPECT_NE(ptr2.data(), nullptr);
  
  ptr2.free();
}

TEST_F(HostDeviceArrayPointerTest, CopyAssignment) {
  auto* manager1 = new chai::expt::HostDeviceArrayManager<int>(5);
  HostDeviceArrayPointer<int> ptr1(manager1);
  
  auto* manager2 = new chai::expt::HostDeviceArrayManager<int>(10);
  HostDeviceArrayPointer<int> ptr2(manager2);

  ptr2.free();
  ptr2 = ptr1;
  
  EXPECT_EQ(ptr2.size(), 5);
  EXPECT_NE(ptr2.data(), nullptr);
  
  ptr2.free();
}

TEST_F(HostDeviceArrayPointerTest, NullptrAssignment) {
  auto* manager = new chai::expt::HostDeviceArrayManager<int>(5);
  HostDeviceArrayPointer<int> ptr(manager);
  
  ptr = nullptr;
  
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.data(), nullptr);
  
  delete manager;
}

TEST_F(HostDeviceArrayPointerTest, Resize) {
  HostDeviceArrayPointer<int> ptr;
  
  ptr.resize(10);
  EXPECT_EQ(ptr.size(), 10);
  EXPECT_NE(ptr.data(), nullptr);
  
  ptr.resize(5);
  EXPECT_EQ(ptr.size(), 5);
  EXPECT_NE(ptr.data(), nullptr);

  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, Free) {
  HostDeviceArrayPointer<int> ptr;
  
  ptr.resize(10);
  ptr.free();
  EXPECT_EQ(ptr.size(), 0);
  EXPECT_EQ(ptr.data(), nullptr);
}

TEST_F(HostDeviceArrayPointerTest, DataAccess) {
  HostDeviceArrayPointer<int> ptr;
  ptr.resize(5);
  
  auto* data = ptr.data();
  EXPECT_NE(data, nullptr);
  
  // Test cdata() too
  auto* cdata = ptr.cdata();
  EXPECT_NE(cdata, nullptr);
  
  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, Update) {
  auto* manager = new chai::expt::HostDeviceArrayManager<int>(5);
  HostDeviceArrayPointer<int> ptr(manager);
  
  manager->resize(10);
  EXPECT_EQ(ptr.size(), 5); // Size hasn't been updated yet
  
  ptr.update();
  EXPECT_EQ(ptr.size(), 10); // Now size should be updated
  
  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, ElementAccess) {
  HostDeviceArrayPointer<int>  ptr;
  ptr.resize(5);
  
  // Initialize array
  for (std::size_t i = 0; i < ptr.size(); ++i) {
    ptr.set(i, static_cast<int>(i * 10));
  }
  
  // Test get()
  for (std::size_t i = 0; i < ptr.size(); ++i) {
    EXPECT_EQ(ptr.get(i), static_cast<int>(i * 10));
  }

  // Test operator[]
  ptr.update();

  for (std::size_t i = 0; i < ptr.size(); ++i) {
    EXPECT_EQ(ptr[i], static_cast<int>(i * 10));
  }
  
  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, GetSetMethods) {
  HostDeviceArrayPointer<int> ptr;
  ptr.resize(5);
  
  // Test set()
  ptr.set(2, 42);
  
  // Test get()
  EXPECT_EQ(ptr.get(2), 42);
  
  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, ExceptionHandling) {
  HostDeviceArrayPointer<int> ptr;

  // Test out-of-bounds access with no underlying array
  EXPECT_THROW(ptr.get(0), std::out_of_range);
  EXPECT_THROW(ptr.set(0, 42), std::out_of_range);

  // Now allocate memory
  ptr.resize(5);
  
  // Test out-of-bounds access with get()
  EXPECT_THROW(ptr.get(10), std::out_of_range);
  
  // Test out-of-bounds access with set()
  EXPECT_THROW(ptr.set(10, 42), std::out_of_range);
  
  ptr.free();
}

TEST_F(HostDeviceArrayPointerTest, LambdaCapture) {
  HostDeviceArrayPointer<int> ptr;
  ptr.resize(5);

  // Initialize array
  auto f = [=] (std::size_t i) { ptr[i] = i; };

  for (std::size_t i = 0; i < ptr.size(); ++i) {
    f(i);
  }

  ptr.free();
}

} // namespace