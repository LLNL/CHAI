// ---------------------------------------------------------------------
// Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
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
#include <climits>

#include "benchmark/benchmark.h"

#include "chai/config.hpp"
#include "chai/managed_ptr.hpp"

#include "../src/util/forall.hpp"

class Base {
   public:
      CHAI_HOST_DEVICE virtual int getValue() const = 0;
};

class Derived : public Base {
   public:
      CHAI_HOST_DEVICE Derived(int value) : Base(), m_value(value) {}

      CHAI_HOST_DEVICE int getValue() const override { return m_value; }

   private:
      int m_value = -1;
};

template <typename T>
class BaseCRTP {
   public:
      CHAI_HOST_DEVICE int getValue() const {
         return static_cast<const T*>(this)->getValue();
      }
};

class DerivedCRTP : public BaseCRTP<DerivedCRTP> {
   public:
      CHAI_HOST_DEVICE DerivedCRTP(int value) : BaseCRTP<DerivedCRTP>(), m_value(value) {}

      CHAI_HOST_DEVICE int getValue() const { return m_value; }

   private:
      int m_value = -1;
};

class NoInheritance {
   public:
      CHAI_HOST_DEVICE NoInheritance(int value) : m_value(value) {}

      CHAI_HOST_DEVICE int getValue() const { return m_value; }

   private:
      int m_value = -1;
};

template <size_t N>
class ClassWithSize {
   private:
      char m_values[N];
};

static void benchmark_managed_ptr_construction_and_destruction(benchmark::State& state)
{
  while (state.KeepRunning()) {
    chai::managed_ptr<Base> temp = chai::make_managed<Derived>(1);
    temp.free();
  }
}

BENCHMARK(benchmark_managed_ptr_construction_and_destruction);

// managed_ptr
static void benchmark_managed_ptr_use_cpu(benchmark::State& state)
{
  chai::managed_ptr<Base> helper = chai::make_managed<Derived>(1);

  while (state.KeepRunning()) {
    forall(sequential(), 0, 1, [=] (int i) { (void) helper->getValue(); });
  }

  helper.free();
}

BENCHMARK(benchmark_managed_ptr_use_cpu);

// Curiously recurring template pattern
static void benchmark_curiously_recurring_template_pattern_cpu(benchmark::State& state)
{
  BaseCRTP<DerivedCRTP>* helper = new DerivedCRTP(3);

  while (state.KeepRunning()) {
    forall(sequential(), 0, 1, [=] (int i) { (void) helper->getValue(); });
  }

  delete helper;
}

BENCHMARK(benchmark_curiously_recurring_template_pattern_cpu);

// Class without inheritance
static void benchmark_no_inheritance_cpu(benchmark::State& state)
{
  NoInheritance* helper = new NoInheritance(5);

  while (state.KeepRunning()) {
    forall(sequential(), 0, 1, [=] (int i) { (void) helper->getValue(); });
  }

  delete helper;
}

BENCHMARK(benchmark_no_inheritance_cpu);

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

template <size_t N>
__global__ void copy_kernel(ClassWithSize<N>) {}

// Benchmark how long it takes to copy a class to the GPU
void benchmark_pass_copy_to_gpu_8(benchmark::State& state)
{
  ClassWithSize<8> helper;

  while (state.KeepRunning()) {
    copy_kernel<<<1, 1>>>(helper);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_pass_copy_to_gpu_8);

void benchmark_pass_copy_to_gpu_64(benchmark::State& state)
{
  ClassWithSize<64> helper;

  while (state.KeepRunning()) {
    copy_kernel<<<1, 1>>>(helper);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_pass_copy_to_gpu_64);

void benchmark_pass_copy_to_gpu_512(benchmark::State& state)
{
  ClassWithSize<512> helper;

  while (state.KeepRunning()) {
    copy_kernel<<<1, 1>>>(helper);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_pass_copy_to_gpu_512);

void benchmark_pass_copy_to_gpu_4096(benchmark::State& state)
{
  ClassWithSize<4096> helper;

  while (state.KeepRunning()) {
    copy_kernel<<<1, 1>>>(helper);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_pass_copy_to_gpu_4096);

void benchmark_managed_ptr_use_gpu(benchmark::State& state)
{
  chai::managed_ptr<Base> helper = chai::make_managed<Derived>(2);

  while (state.KeepRunning()) {
    forall(gpu(), 0, 1, [=] __device__ (int i) { (void) helper->getValue(); });
  }

  helper.free();
}

// Benchmark how long it takes to call placement new on the GPU
template <size_t N>
__global__ void placement_new_kernel(ClassWithSize<N>* address) {
   (void) new(address) ClassWithSize<N>();
}

template <size_t N>
__global__ void placement_delete_kernel(ClassWithSize<N>* address) {
   address->~ClassWithSize<N>();
}

void benchmark_placement_new_on_gpu_8(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<8>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<8>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_8);

void benchmark_placement_new_on_gpu_64(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<64>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<64>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_64);

void benchmark_placement_new_on_gpu_512(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<512>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<512>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_512);

void benchmark_placement_new_on_gpu_4096(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<4096>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<4096>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_4096);

void benchmark_placement_new_on_gpu_32768(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<32768>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<32768>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_32768);

void benchmark_placement_new_on_gpu_262144(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<262144>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<262144>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_262144);

void benchmark_placement_new_on_gpu_2097152(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<2097152>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<2097152>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_placement_new_on_gpu_2097152);

// Benchmark how long it takes to call new on the GPU
template <size_t N>
__global__ void create_kernel(ClassWithSize<N>** address) {
   *address = new ClassWithSize<N>();
}

template <size_t N>
__global__ void delete_kernel(ClassWithSize<N>** address) {
   delete *address;
}

void benchmark_new_on_gpu_8(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<8>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<8>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_8);

void benchmark_new_on_gpu_64(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<64>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<64>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_64);

void benchmark_new_on_gpu_512(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<512>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<512>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_512);

void benchmark_new_on_gpu_4096(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<4096>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<4096>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_4096);

void benchmark_new_on_gpu_32768(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<32768>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<32768>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_32768);

void benchmark_new_on_gpu_262144(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<262144>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<262144>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_262144);

void benchmark_new_on_gpu_2097152(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<2097152>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<2097152>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_2097152);

// Benchmark current approach
template <size_t N>
__global__ void delete_kernel_2(ClassWithSize<N>* address) {
   delete address;
}

void benchmark_new_on_gpu_and_copy_to_host_8(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<8>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<8>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<8>** cpuBuffer = (ClassWithSize<8>**) malloc(sizeof(ClassWithSize<8>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<8>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<8>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_8);

void benchmark_new_on_gpu_and_copy_to_host_64(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<64>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<64>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<64>** cpuBuffer = (ClassWithSize<64>**) malloc(sizeof(ClassWithSize<64>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<64>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<64>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_64);

void benchmark_new_on_gpu_and_copy_to_host_512(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<512>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<512>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<512>** cpuBuffer = (ClassWithSize<512>**) malloc(sizeof(ClassWithSize<512>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<512>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<512>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_512);

void benchmark_new_on_gpu_and_copy_to_host_4096(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<4096>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<4096>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<4096>** cpuBuffer = (ClassWithSize<4096>**) malloc(sizeof(ClassWithSize<4096>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<4096>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<4096>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_4096);

void benchmark_new_on_gpu_and_copy_to_host_32768(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<32768>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<32768>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<32768>** cpuBuffer = (ClassWithSize<32768>**) malloc(sizeof(ClassWithSize<32768>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<32768>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<32768>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_32768);

void benchmark_new_on_gpu_and_copy_to_host_262144(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<262144>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<262144>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<262144>** cpuBuffer = (ClassWithSize<262144>**) malloc(sizeof(ClassWithSize<262144>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<262144>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<262144>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_262144);

void benchmark_new_on_gpu_and_copy_to_host_2097152(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<2097152>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<2097152>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<2097152>** cpuBuffer = (ClassWithSize<2097152>**) malloc(sizeof(ClassWithSize<2097152>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<2097152>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<2097152>* gpuPointer = cpuBuffer[0];
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    free(cpuBuffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_new_on_gpu_and_copy_to_host_2097152);

// Benchmark how long it takes to create a stack object on the GPU
template <size_t N>
__global__ void create_on_stack_kernel() {
   (void) ClassWithSize<N>();
}

void benchmark_create_on_stack_on_gpu_8(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<8><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_8);

void benchmark_create_on_stack_on_gpu_64(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<64><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_64);

void benchmark_create_on_stack_on_gpu_512(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<512><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_512);

void benchmark_create_on_stack_on_gpu_4096(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<4096><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_4096);

void benchmark_create_on_stack_on_gpu_32768(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<32768><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_32768);

void benchmark_create_on_stack_on_gpu_262144(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<262144><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_262144);

void benchmark_create_on_stack_on_gpu_2097152(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<2097152><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK(benchmark_create_on_stack_on_gpu_2097152);

BENCHMARK(benchmark_managed_ptr_use_gpu);

// Curiously recurring template pattern
void benchmark_curiously_recurring_template_pattern_gpu(benchmark::State& state)
{
  BaseCRTP<DerivedCRTP>* derivedCRTP = new DerivedCRTP(4);
  auto helper = *derivedCRTP;

  while (state.KeepRunning()) {
    forall(gpu(), 0, 1, [=] __device__ (int i) { (void) helper.getValue(); });
  }

  delete derivedCRTP;
}

BENCHMARK(benchmark_curiously_recurring_template_pattern_gpu);

// Class without inheritance
void benchmark_no_inheritance_gpu(benchmark::State& state)
{
  NoInheritance* noInheritance = new NoInheritance(5);
  auto helper = *noInheritance;

  while (state.KeepRunning()) {
    forall(gpu(), 0, 1, [=] __device__ (int i) { (void) helper.getValue(); });
  }

  delete noInheritance;
}

BENCHMARK(benchmark_no_inheritance_gpu);

#endif

BENCHMARK_MAIN();
