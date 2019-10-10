#include "camp/device.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

int main()
{
  std::cout << "Chai Context Implementation\n";

  //camp::devices::Context cuda_context{camp::devices::Cuda()};
  //camp::devices::Context host_context{camp::devices::Host()};

  std::vector< chai::ManagedArray<float> > arrays(1);


  std::cout << "calling forall with cuda context" << std::endl;
  for (auto array : arrays) {
    auto lambda_set = [=] CHAI_HOST_DEVICE (int i) { array[i] = i; };
    camp::devices::Context ctx{camp::devices::Cuda()};
    array.allocate(10);
    auto e = forall(&ctx, 0, 10, lambda_set);
    array.move(chai::CPU, &ctx);
  }

  std::cout << "calling forall with host context" << std::endl;
  for (auto array : arrays) {
    auto lambda_check = [=] CHAI_HOST_DEVICE (int i) { array[i] = 123; };
    camp::devices::Context ctx{camp::devices::Host{}}; 
    auto e = forall(&ctx, 0, 10, lambda_check);
  }

  for (auto array : arrays) {
    std::cout<< array[0] << std::endl;
  }

  for (auto a : arrays) a.free();
  return 0;
}
