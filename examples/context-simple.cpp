#include "camp/resources.hpp"
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
  constexpr std::size_t ARRAY_SIZE{100};
  std::vector<chai::ManagedArray<double>> arrays;
  camp::resources::Context host{camp::resources::Host{}}; 

  int clockrate{get_clockrate()};

  for (std::size_t i = 0; i < 10; ++i) {
    arrays.push_back(chai::ManagedArray<double>(ARRAY_SIZE));
  }

  for (auto array : arrays) {
    // set on host
    forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
        array[i] = i;
    }); 
  }

  for (auto array : arrays) {
    camp::resources::Context context{camp::resources::Cuda{}}; 

    forall(&context, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
        array[i] = array[i] * 2.0;
        wait_for(20, clockrate);
    });

    array.move(chai::CPU, &context);
  }

  for (auto array : arrays) {
    forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
        if (i == 256) {
          printf("array[%d] = %f \n", i, array[i]);
        }
    }); 
  }
}
