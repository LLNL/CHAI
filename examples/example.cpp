#include "chai/ManagedArray.hpp"

int main(int argc, char* argv[]) {
  chai::ManagedArray<float> array(10);

  forall<sequential>(0, 10, [=] (int i) {
      array[i] = static_cast<float>(i);
  });

  forall<sequential>(0, 10, [=] (int i) {
      std::cout << "array[ " << i << "] = " << array[i] << std::endl;
  });

  forall<cuda>(0, 10, [=] __device__ (int i) {
      array[i] *= 2.0f;
  });

  forall<sequential>(0, 10, [=] (int i) {
      std::cout << "array[ " << i << "] = " << array[i] << std::endl;
  });
}
