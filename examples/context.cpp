#include "camp/device.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

int main()
{
  std::cout << "Chai Context Implementation\n";

  camp::devices::Context cuda_context{camp::devices::Cua()}; 
  camp::devices::Context host_context{camp::devices::Host()}; 

  chai::ManagedArray<float> array(10);

  std::cout << "defining lambda" << std::endl;
  auto lambda_set = [=] CHAI_HOST_DEVICE (int i) { array[i] = i; };
  auto lambda_check = [=] CHAI_HOST_DEVICE (int i) { array[i] = 123; };

  std::cout << "calling forall with cuda context" << std::endl;
  auto e = forall(&cuda_context, 0, 10, lambda_set);
 
  e.wait();
  std::cout << "calling forall with host context" << std::endl;
  forall(&host_context, 0, 10, lambda_check);

  array.free();
  return 0;
}
