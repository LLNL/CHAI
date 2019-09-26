#include "camp/device.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

int main()
{
  std::cout << "Chai Context Implementation\n";

  camp::devices::Context res_context{camp::devices::Host()}; 

  chai::ManagedArray<float> array(10);

  std::cout << "defining lambda" << std::endl;
  auto lambda = [=] CHAI_HOST_DEVICE (int i) { array[i] = i; };

  std::cout << "calling forall with context" << std::endl;
  forall(&res_context, 0, 10, lambda);

  array.free();
  return 0;
}
