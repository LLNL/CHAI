//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_forall_HPP
#define CHAI_forall_HPP

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/config.hpp"
//#include "camp/device.hpp"
#include "camp/resources.hpp"

#if defined(CHAI_ENABLE_UM)
#include <cuda_runtime_api.h>
#endif

struct sequential {
};
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
struct gpu {
};
#endif

template <typename LOOP_BODY>
void forall_kernel_cpu(int begin, int end, LOOP_BODY body)
{
  for (int i = 0; i < (end - begin); ++i) {
    body(i);
  }
}

/*
 * \brief Run forall kernel on CPU.
 */
template <typename LOOP_BODY>
void forall(sequential, int begin, int end, LOOP_BODY body)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

#if defined(CHAI_ENABLE_UM)
  cudaDeviceSynchronize();
#endif

  rm->setExecutionSpace(chai::CPU);

  forall_kernel_cpu(begin, end, body);

  rm->setExecutionSpace(chai::NONE);
}
template <typename LOOP_BODY>
camp::resources::Event forall_host(camp::resources::Context* dev, int begin, int end, LOOP_BODY body)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

#if defined(CHAI_ENABLE_UM)
  cudaDeviceSynchronize();
#endif

  rm->setExecutionSpace(chai::CPU, dev);

  auto host = dev->get<camp::resources::Host>();
  forall_kernel_cpu(begin, end, body);

  rm->setExecutionSpace(chai::NONE);
  return dev->get_event();
}



#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
template <typename LOOP_BODY>
__global__ void forall_kernel_gpu(int start, int length, LOOP_BODY body)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx);
  }
}

/*
 * \brief Run forall kernel on GPU.
 */
template <typename LOOP_BODY>
void forall(gpu, int begin, int end, LOOP_BODY&& body)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

  rm->setExecutionSpace(chai::GPU);

  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1) / blockSize;

#if defined(CHAI_ENABLE_CUDA)
  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end - begin, body);
  cudaDeviceSynchronize();
#elif defined(CHAI_ENABLE_HIP)
  hipLaunchKernelGGL(forall_kernel_gpu, dim3(gridSize), dim3(blockSize), 0,0,
                     begin, end - begin, body);
  hipDeviceSynchronize();
#endif
  
  rm->setExecutionSpace(chai::NONE);
}
template <typename LOOP_BODY>
camp::resources::Event forall_gpu(camp::resources::Context* dev, int begin, int end, LOOP_BODY&& body)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

  rm->setExecutionSpace(chai::GPU, dev);

  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1) / blockSize;

#if defined(CHAI_ENABLE_CUDA)
auto cuda = dev->get<camp::resources::Cuda>();
forall_kernel_gpu<<<gridSize, blockSize, 0, cuda.get_stream()>>>(begin, end - begin, body);
#elif defined(CHAI_ENABLE_HIP)
  hipLaunchKernelGGL(forall_kernel_gpu, dim3(gridSize), dim3(blockSize), 0,0,
                     begin, end - begin, body);
  hipDeviceSynchronize();
#endif
  
  rm->setExecutionSpace(chai::NONE);
  return dev->get_event();
}
#endif // if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

template <typename LOOP_BODY>
camp::resources::Event forall(camp::resources::Context *con, int begin, int end, LOOP_BODY&& body)
{
  auto platform = con->get_platform();
  switch(platform) {
    case camp::resources::Platform::cuda:
    case camp::resources::Platform::hip:
	return forall_gpu(con, begin, end, body);
    default:
	return forall_host(con, begin, end, body);
  }
}

#endif  // CHAI_forall_HPP
