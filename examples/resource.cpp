#include "camp/resource.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"
#include <cuda_profiler_api.h>

int main()
{
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

  const int NUM_ARRAYS = 16;
  const int ARRAY_SIZE = 10;
  std::vector< chai::ManagedArray<float> > arrays;

  auto callBack = [&](const chai::PointerRecord* record, chai::Action act, chai::ExecutionSpace s)
  {
    const size_t bytes = record->m_size;
    printf("%s cback: act=%s, space=%s, bytes=%ld\n", record->name.c_str(), chai::PrintAction[(int) act], chai::PrintExecSpace[(int) s], (long) bytes);
    if (act == chai::ACTION_MOVE)
    {
      if (s == chai::CPU)
      {
        printf("Moved to host\n");
      }
      else if (s == chai::GPU)
      {
        printf("Moved to device\n");
      }
    }
    if (act == chai::ACTION_FOUND_ABANDONED) {
       printf("in abandoned!\n");
       //ASSERT_EQ(false,true);
    }
  };

  for (int i = 0; i < NUM_ARRAYS; i++) {
    arrays.push_back(chai::ManagedArray<float>(10, chai::GPU));
    arrays[i].m_pointer_record->name = "array "+ std::to_string(i);
    arrays[i].setUserCallback(callBack);
  }

  std::cout << "calling forall with cuda context" << std::endl;
  for (auto array : arrays) {

    camp::resources::Resource res{camp::resources::Cuda()};

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


    std::cout << "Calling forall" << std::endl;
    auto e = forall(&res, 0, ARRAY_SIZE, clock_lambda_1);
    std::cout << "Move to CPU called" << std::endl;
    array.move(chai::CPU, &res); // asynchronous move
  }

  std::cout << "calling forall with host context" << std::endl;
  for (auto array : arrays) {
    auto clock_lambda_2 = [=] CHAI_HOST_DEVICE (int idx) {
      array[idx] *= array[idx];
    };
    camp::resources::Resource res{camp::resources::Host{}}; 
    auto e = forall(&res, 0, ARRAY_SIZE, clock_lambda_2);
  }

  camp::resources::Resource host{camp::resources::Host{}};
  for (auto array : arrays) {
    forall(&host, 0, 10, [=] CHAI_HOST_DEVICE (int i) {
      printf("%i ", int(array[i]) );
    });
    printf("\n");
  }

  for (auto a : arrays) a.free();
  return 0;
}
