#ifndef forall_HPP
#define forall_HPP

#include "chai/ExecutionSpaces.hpp"
#include "chai/ResourceManager.hpp"

struct sequential {};
struct cuda {};

template <typename LOOP_BODY>
void forall_kernel_cpu(int begin, int end, LOOP_BODY body)
{
  for (int i = 0; i < (end - begin); ++i) {
    body(i);
  }
}

template <typename LOOP_BODY>
void forall(sequential, int begin, int end, LOOP_BODY body) {
  chai::ResourceManager* rm = chai::ResourceManager::getResourceManager();

  rm->setExecutionSpace(chai::CPU);

  forall_kernel_cpu(begin, end, body);

  rm->setExecutionSpace(chai::NONE);
}

template <typename LOOP_BODY>
__global__ void forall_kernel_gpu(int start, int length, LOOP_BODY body) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx);
  }
}

template <typename LOOP_BODY>
void forall(cuda, int begin, int end, LOOP_BODY&& body) {
  chai::ResourceManager* rm = chai::ResourceManager::getResourceManager();

  rm->setExecutionSpace(chai::GPU);

  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1)/blockSize;

  forall_kernel_gpu<<<gridSize, blockSize>>>(begin, end-begin, body);

  rm->setExecutionSpace(chai::NONE);
}

#endif
