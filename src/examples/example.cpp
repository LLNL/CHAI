#include "chai/ManagedArray.hpp"

#include "forall.hpp"

#include <iostream>

int main(int argc, char* argv[]) {

  chai::ManagedArray<float> v1(10);
  chai::ManagedArray<float> v2(10);

  /*
   * Allocate an array on the device only
   */
  chai::ManagedArray<int> i1(10, chai::GPU);

  std::cout << "Created new array..." << std::endl;

  forall(sequential(), 0, 10, [=] (int i) {
      v1[i] = static_cast<float>(i * 1.0f);
  });

  std::cout << "v1 = [";
  forall(sequential(), 0, 10, [=] (int i) {
      std::cout << " " << v1[i] << std::endl;
  });
  std::cout << "]" << std::endl;

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      v2[i] = v1[i]*2.0f;
  });

  std::cout << "v2 = [";
  forall(sequential(), 0, 10, [=] (int i) {
      std::cout << " " << v2[i] << std::endl;
  });
  std::cout << "]" << std::endl;

  forall(cuda(), 0, 10, [=] __device__ (int i) {
      v2[i] *= 2.0f;
  });

  float * raw_v2 = v2;

  std::cout << "raw_v2 = [";
  for (int i = 0; i < 10; i++ ) {
      std::cout << " " << raw_v2[i] << std::endl;
  }
  std::cout << "]" << std::endl;
}
