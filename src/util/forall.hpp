//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_forall_HPP
#define CHAI_forall_HPP

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/config.hpp"

#if defined(CHAI_ENABLE_UM)
#include <cuda_runtime_api.h>
#endif

struct sequential {
};
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
struct gpu {
};

struct gpu_async {};
#endif

template <typename LOOP_BODY>
void forall_kernel_cpu(int begin, int end, LOOP_BODY body)
{
  for (int i = begin; i < end; ++i) {
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

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
template <typename LOOP_BODY>
__global__ void forall_kernel_gpu(int start, int length, LOOP_BODY body)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx+start);
  }
}

template <typename LOOP_BODY>
void forall(gpu_async, int begin, int end, LOOP_BODY&& body)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

  rm->setExecutionSpace(chai::GPU);

#if defined(CHAI_ENABLE_CUDA)
  size_t blockSize = 32;
#elif defined(CHAI_ENABLE_HIP)
  size_t blockSize = 64;
#endif

  size_t gridSize = (end - begin + blockSize - 1) / blockSize;

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  forall_kernel_cpu(begin, end, body);
#elif defined(CHAI_ENABLE_CUDA)
  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end - begin, body);
#elif defined(CHAI_ENABLE_HIP)
  hipLaunchKernelGGL(forall_kernel_gpu, dim3(gridSize), dim3(blockSize), 0, 0,
                     begin, end - begin, body);
#endif
  rm->setExecutionSpace(chai::NONE);
}

/*
 * \brief Run forall kernel on GPU.
 */
template <typename LOOP_BODY>
void forall(gpu, int begin, int end, LOOP_BODY&& body)
{
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

  rm->setExecutionSpace(chai::GPU);

#if defined(CHAI_ENABLE_CUDA)
  size_t blockSize = 32;
#elif defined(CHAI_ENABLE_HIP)
  size_t blockSize = 64;
#endif

  size_t gridSize = (end - begin + blockSize - 1) / blockSize;

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  forall_kernel_cpu(begin, end, body);
#elif defined(CHAI_ENABLE_CUDA)
  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end - begin, body);
  cudaDeviceSynchronize();
#elif defined(CHAI_ENABLE_HIP)
  hipLaunchKernelGGL(forall_kernel_gpu, dim3(gridSize), dim3(blockSize), 0, 0,
                     begin, end - begin, body);
  hipDeviceSynchronize();
#endif
  
  rm->setExecutionSpace(chai::NONE);
}
#endif

#endif  // CHAI_forall_HPP
