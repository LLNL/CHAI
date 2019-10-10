#include "camp/device.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"
#include <cuda_profiler_api.h>

int main()
{
  std::cout << "Chai Context Implementation\n";


  float kernel_time = 20;
  int cuda_device = 0;

  cudaDeviceProp deviceProp;
  cudaGetDevice(&cuda_device);
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  if ((deviceProp.concurrentKernels == 0))
  {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }
  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
   deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

#if defined(__arm__) || defined(__aarch64__)
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif



  const int NUM_ARRAYS = 2;
  const int ARRAY_SIZE = 10;
  std::vector< chai::ManagedArray<float> > arrays;
  std::vector< camp::devices::Context > cuda_ctx;
  for (int i = 0; i < NUM_ARRAYS; i++) {
    arrays.push_back(chai::ManagedArray<float>(10, chai::GPU));
    cuda_ctx.push_back(camp::devices::Context{camp::devices::Cuda()});
  }

  std::cout << "calling forall with cuda context" << std::endl;
  for (auto array : arrays) {
  //for (int i = 0; i < NUM_ARRAYS; i++) {
  //  auto array = arrays[i]; auto ctx = &cuda_ctx[i];

    camp::devices::Context ctx{camp::devices::Cuda()};
    auto clock_lambda_1 = [=] CHAI_HOST_DEVICE (int idx) {
      array[idx] = idx * 2;
      unsigned int start_clock = (unsigned int) clock();
      clock_t clock_offset = 0;
      while (clock_offset < time_clocks)
      {
	unsigned int end_clock = (unsigned int) clock();
	clock_offset = (clock_t)(end_clock - start_clock);
      }
    };

    auto e = forall(&ctx, 0, ARRAY_SIZE, clock_lambda_1);
    array.move(chai::CPU, &ctx); // asynchronous move
  }


  //for (int i = 0; i < NUM_ARRAYS; i++) {
  //  auto array = arrays[i]; auto ctx = &cuda_ctx[i];
  //  array.move(chai::CPU, ctx); // asynchronous move
  //}

  std::cout << "calling forall with host context" << std::endl;
  for (auto array : arrays) {
    auto clock_lambda_2 = [=] CHAI_HOST_DEVICE (int idx) {
      array[idx] *= array[idx];
    };
    camp::devices::Context ctx{camp::devices::Host{}}; 
    auto e = forall(&ctx, 0, ARRAY_SIZE, clock_lambda_2);
  }

  std::cout << "printing..." << std::endl;
  for (auto array : arrays) {
    for (int i = 0; i < ARRAY_SIZE; i++) {
      std::cout<< array[i] << " ";
    }
    std::cout << std::endl;
  }

  for (auto a : arrays) a.free();
  return 0;
}
