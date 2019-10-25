#include "camp/contexts.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

#include <vector>
#include <utility>

inline __host__ __device__ void
wait_for(float time, float clockrate) {
  clock_t time_in_clocks = time*clockrate;

  unsigned int start_clock = (unsigned int) clock();
  clock_t clock_offset = 0;
  while (clock_offset < time_in_clocks)
  {
    unsigned int end_clock = (unsigned int) clock();
    clock_offset = (clock_t)(end_clock - start_clock);
  }
}

int get_clockrate()
{
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
  return deviceProp.clockRate/1000;
#else
  return deviceProp.clockRate;
#endif
}

int main()
{
  constexpr std::size_t ARRAY_SIZE{1000};
  int clockrate{get_clockrate()}; 

  chai::ManagedArray<double> array1(ARRAY_SIZE);
  chai::ManagedArray<double> array2(ARRAY_SIZE);

  camp::resources::Context dev1{camp::resources::Cuda{}};
  camp::resources::Context dev2{camp::resources::Cuda{}};

  auto e1 = forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array1[i] = i;
      wait_for(10, clockrate);
  });

  auto e2 = forall(&dev2, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array2[i] = -1;
      wait_for(20, clockrate);
  });

  e2.wait();
  e1.wait();

  forall(&dev1, 0, ARRAY_SIZE, [=] CHAI_HOST_DEVICE (int i) {
      array1[i] *= array2[i];
      wait_for(10, clockrate);
  });

  array1.move(chai::CPU, &dev1);

  camp::resources::Context host{camp::resources::Host{}};

  forall(&host, 0, 10, [=] CHAI_HOST_DEVICE (int i) {
      printf("%f ", array1[i]);
  });
  printf("\n");
}
