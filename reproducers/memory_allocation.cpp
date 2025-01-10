#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "chai/config.hpp"
#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"

#include <iostream>
#include <chrono>

/************************************************************************
* Initial sections is just a bunch of helper code used for later steps
*/

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define __test_host__   __host__
#define __test_device__ __device__
#define __test_global__ __global__
#define __test_hdev__   __host__ __device__
#define __test_gpu_active__
#else
#define __test_host__
#define __test_device__
#define __test_global__
#define __test_hdev__
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))  || defined(__HIP_DEVICE_COMPILE__)
#define __test_device_only__      
#else
#define __test_host_only__      
#endif

#define NUMBLOCKS 256
#define ASYNC false

struct MemoryManager {

  MemoryManager() : rm(umpire::ResourceManager::getInstance()) {
    device_allocator = rm.makeAllocator<umpire::strategy::QuickPool>
      ("SNLS_DEVICE_pool", rm.getAllocator("DEVICE"),
       1024 * 1024 * 1024);
  }

  template<typename T>
  __test_host__
  inline
  chai::ManagedArray<T> allocManagedArray(std::size_t size=0)
  {
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
    auto es = chai::ExecutionSpace::GPU;
#else
    auto es = chai::ExecutionSpace::CPU;
#endif

    chai::ManagedArray<T> array(size, 
				std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
				    , chai::GPU
#endif
				    },
				std::initializer_list<umpire::Allocator>{rm.getAllocator("HOST")
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
				    , device_allocator
#endif
				    },
				es
				);

    return array;
  }

  umpire::ResourceManager& rm;
  umpire::Allocator device_allocator;
};

int main() {

  constexpr size_t npoints = 6000000;
  constexpr size_t nitems_per_point = 40; // 40 doubles (40 x 64 bytes)
  constexpr long nbytes = npoints * nitems_per_point * 64ul;

  constexpr size_t nsteps = 100;

  double total_time = 0.0;

  MemoryManager mm;

  for (size_t i = 0; i < nsteps; i++ ) {
      std::chrono::time_point<std::chrono::system_clock> start, end;  
      start = std::chrono::system_clock::now();

#ifdef JUST_HIP_MALLOC // for testing purposes
      void* mem;
      hipMalloc(&mem, nbytes);
#else
      auto array = mm.allocManagedArray<double>(npoints * nitems_per_point);
#endif 
      end = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = end-start;
      total_time += elapsed_seconds.count();

#ifdef JUST_HIP_MALLOC
      hipFree(&mem);
#else
      array.free();
#endif
  }

  std::cout << "total time: " << total_time << std::endl;
  
  return 0;

}
