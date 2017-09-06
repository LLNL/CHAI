// ---------------------------------------------------------------------
// Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
// rights reserved.
// 
// Produced at the Lawrence Livermore National Laboratory.
// 
// This file is part of CHAI.
// 
// LLNL-CODE-705877
// 
// For details, see https:://github.com/LLNL/CHAI
// Please also see the NOTICE and LICENSE files.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the
//   distribution.
// 
// - Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------------
#ifndef CHAI_forall_HPP
#define CHAI_forall_HPP

#include "chai/config.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/ArrayManager.hpp"

#if defined(CHAI_ENABLE_THIN_UM)
#include <cuda_runtime_api.h>
#endif

struct sequential {};
#if defined(CHAI_ENABLE_CUDA)
struct cuda {};
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
void forall(sequential, int begin, int end, LOOP_BODY body) {
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

#if defined(CHAI_ENABLE_THIN_UM)
  cudaDeviceSynchronize();
#endif

  rm->setExecutionSpace(chai::CPU);

  forall_kernel_cpu(begin, end, body);

  rm->setExecutionSpace(chai::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
template <typename LOOP_BODY>
__global__ void forall_kernel_gpu(int start, int length, LOOP_BODY body) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx);
  }
}

/*
 * \brief Run forall kernel on GPU.
 */
template <typename LOOP_BODY>
void forall(cuda, int begin, int end, LOOP_BODY&& body) {
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();

  rm->setExecutionSpace(chai::GPU);

  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1)/blockSize;

  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end-begin, body);

  rm->setExecutionSpace(chai::NONE);
}
#endif

#endif // CHAI_forall_HPP
