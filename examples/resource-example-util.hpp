#ifndef RESOURCE_EXAMPLE_UTIL_HPP
#define RESOURCE_EXAMPLE_UTIL_HPP


inline __host__ __device__ void
gpu_time_wait_for(float time, float clockrate) {
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

#endif  // RESOURCE_EXAMPLE_UTIL_HPP
