#ifndef CHAI_DEVICE_HELPERS_HPP
#define CHAI_DEVICE_HELPERS_HPP

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"

namespace chai
{
// CHAI_GPU_ERROR_CHECK macro
#ifdef CHAI_ENABLE_DEVICE

#ifdef CHAI_ENABLE_GPU_ERROR_CHECKING

inline void gpuErrorCheck(gpuError_t code, const char *file, int line, bool abort=true)
{
   if (code != gpuSuccess) {
      fprintf(stderr, "[CHAI] GPU Error: %s %s %d\n", gpuGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}

#define CHAI_GPU_ERROR_CHECK(code) { ::chai::gpuErrorCheck((code), __FILE__, __LINE__); }
#else // CHAI_ENABLE_GPU_ERROR_CHECKING
#define CHAI_GPU_ERROR_CHECK(code) code
#endif // CHAI_ENABLE_GPU_ERROR_CHECKING

#endif

// wrapper for hip/cuda synchronize
inline void synchronize() {
#if defined(CHAI_ENABLE_DEVICE) && !defined(CHAI_DEVICE_COMPILE)
   CHAI_GPU_ERROR_CHECK(gpuDeviceSynchronize());
#endif
}

#if defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)

// wrapper for hip/cuda free
CHAI_HOST inline void gpuFree(void* buffer) {
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   free(buffer);
#elif defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipFree(buffer));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaFree(buffer));
#endif
}

// wrapper for hip/cuda malloc
CHAI_HOST inline void gpuMalloc(void** devPtr, size_t size) {
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   *devPtr = (void*)malloc(size);
#elif defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipMalloc(devPtr, size));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaMalloc(devPtr, size));
#endif
}

// wrapper for hip/cuda managed malloc
CHAI_HOST inline void gpuMallocManaged(void** devPtr, size_t size) {
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   *devPtr = (void*)malloc(size);
#elif defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipMallocManaged(devPtr, size));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaMallocManaged(devPtr, size));
#endif
}

// wrapper for hip/cuda mem copy
CHAI_HOST inline void  gpuMemcpy(void* dst, const void* src, size_t count, gpuMemcpyKind kind) {
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
   memcpy(dst, src, count);
#elif defined (CHAI_ENABLE_HIP)
   CHAI_GPU_ERROR_CHECK(hipMemcpy(dst, src, count, kind));
#elif defined (CHAI_ENABLE_CUDA)
   CHAI_GPU_ERROR_CHECK(cudaMemcpy(dst, src, count, kind));
#endif
}

#endif //#if defined(CHAI_GPUCC)

}  // namespace chai

#endif // CHAI_DEVICE_HELPERS_HPP
