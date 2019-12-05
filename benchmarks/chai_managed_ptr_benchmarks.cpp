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
      CHAI_HOST_DEVICE virtual void scale(int numValues, int* values) = 0;

      CHAI_HOST_DEVICE virtual void sumAndScale(int numValues, int* values, int& value) = 0;

      CHAI_HOST_DEVICE virtual ~Base() = default;
};

class Derived : public Base {
   public:
      CHAI_HOST_DEVICE Derived(int value) : Base(), m_value(value) {}

      CHAI_HOST_DEVICE virtual void scale(int numValues, int* values) override {
         for (int i = 0; i < numValues; ++i) {
            values[i] *= m_value;
         }
      }

      CHAI_HOST_DEVICE virtual void sumAndScale(int numValues, int* values, int& value) override {
         int result = 0;

         for (int i = 0; i < numValues; ++i) {
            result += values[i];
         }

         value *= m_value * result;
      }

   private:
      int m_value = -1;
};

template <typename T>
class BaseCRTP {
   public:
      CHAI_HOST_DEVICE void scale(int numValues, int* values) {
         return static_cast<T*>(this)->scale(numValues, values);
      }

      CHAI_HOST_DEVICE void sumAndScale(int numValues, int* values, int& value) {
         return static_cast<T*>(this)->sumAndScale(numValues, values, value);
      }
};

class DerivedCRTP : public BaseCRTP<DerivedCRTP> {
   public:
      CHAI_HOST_DEVICE DerivedCRTP(int value) : BaseCRTP<DerivedCRTP>(), m_value(value) {}

      CHAI_HOST_DEVICE void scale(int numValues, int* values) {
         for (int i = 0; i < numValues; ++i) {
            values[i] *= m_value;
         }
      }

      CHAI_HOST_DEVICE void sumAndScale(int numValues, int* values, int& value) {
         int result = 0;

         for (int i = 0; i < numValues; ++i) {
            result += values[i];
         }

         value *= m_value * result;
      }

   private:
      int m_value = -1;
};

class NoInheritance {
   public:
      CHAI_HOST_DEVICE NoInheritance(int value) : m_value(value) {}

      CHAI_HOST_DEVICE void scale(int numValues, int* values) {
         for (int i = 0; i < numValues; ++i) {
            values[i] *= m_value;
         }
      }

      CHAI_HOST_DEVICE void sumAndScale(int numValues, int* values, int& value) {
         int result = 0;

         for (int i = 0; i < numValues; ++i) {
            result += values[i];
         }

         value *= m_value * result;
      }

   private:
      int m_value = -1;
};

template <int N>
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
static void benchmark_use_managed_ptr_cpu(benchmark::State& state)
{
  chai::managed_ptr<Base> object = chai::make_managed<Derived>(2);

  int numValues = 100;
  int* values = (int*) malloc(100 * sizeof(int));

  for (int i = 0; i < numValues; ++i) {
     values[i] = i * i;
  }

#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif

  while (state.KeepRunning()) {
    object->scale(numValues, values);
  }

  object.free();
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
}

BENCHMARK(benchmark_use_managed_ptr_cpu);

// Curiously recurring template pattern
static void benchmark_curiously_recurring_template_pattern_cpu(benchmark::State& state)
{
  BaseCRTP<DerivedCRTP>* object = new DerivedCRTP(2);

  int numValues = 100;
  int* values = (int*) malloc(100 * sizeof(int));

  for (int i = 0; i < numValues; ++i) {
     values[i] = i * i;
  }

  while (state.KeepRunning()) {
    object->scale(numValues, values);
  }

  free(values);
  delete object;
}

BENCHMARK(benchmark_curiously_recurring_template_pattern_cpu);

// Class without inheritance
static void benchmark_no_inheritance_cpu(benchmark::State& state)
{
  NoInheritance* object = new NoInheritance(2);

  int numValues = 100;
  int* values = (int*) malloc(100 * sizeof(int));

  for (int i = 0; i < numValues; ++i) {
     values[i] = i * i;
  }

  while (state.KeepRunning()) {
    object->scale(numValues, values);
  }

  free(values);
  delete object;
}

BENCHMARK(benchmark_no_inheritance_cpu);

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)

template <int N>
__global__ void copy_kernel(ClassWithSize<N>) {}

// Benchmark how long it takes to copy a class to the GPU
template <int N>
static void benchmark_pass_copy_to_gpu(benchmark::State& state)
{
  ClassWithSize<N> helper;

  while (state.KeepRunning()) {
    copy_kernel<<<1, 1>>>(helper);
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(benchmark_pass_copy_to_gpu, 8);
BENCHMARK_TEMPLATE(benchmark_pass_copy_to_gpu, 64);
BENCHMARK_TEMPLATE(benchmark_pass_copy_to_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_pass_copy_to_gpu, 4096);

template <int N>
static void benchmark_copy_to_gpu(benchmark::State& state)
{
  ClassWithSize<N>* cpuPointer = new ClassWithSize<N>();

  while (state.KeepRunning()) {
    ClassWithSize<N>* gpuPointer;
    cudaMalloc(&gpuPointer, sizeof(ClassWithSize<N>));
    cudaMemcpy(gpuPointer, cpuPointer, sizeof(ClassWithSize<N>), cudaMemcpyHostToDevice);
    cudaFree(gpuPointer);
    cudaDeviceSynchronize();
  }

  delete cpuPointer;
}

BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 8);
BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 64);
BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_copy_to_gpu, 2097152);

// Benchmark how long it takes to call placement new on the GPU
template <int N>
__global__ void placement_new_kernel(ClassWithSize<N>* address) {
   (void) new(address) ClassWithSize<N>();
}

template <int N>
__global__ void placement_delete_kernel(ClassWithSize<N>* address) {
   address->~ClassWithSize<N>();
}

template <int N>
static void benchmark_placement_new_on_gpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<N>* address;
    cudaMalloc(&address, sizeof(ClassWithSize<N>));
    placement_new_kernel<<<1, 1>>>(address);
    placement_delete_kernel<<<1, 1>>>(address);
    cudaFree(address);
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 8);
BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 64);
BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_placement_new_on_gpu, 2097152);

// Benchmark how long it takes to call new on the GPU
template <int N>
__global__ void create_kernel(ClassWithSize<N>** address) {
   *address = new ClassWithSize<N>();
}

template <int N>
__global__ void delete_kernel(ClassWithSize<N>** address) {
   delete *address;
}

template <int N>
static void benchmark_new_on_gpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<N>** buffer;
    cudaMalloc(&buffer, sizeof(ClassWithSize<N>*));
    create_kernel<<<1, 1>>>(buffer);
    delete_kernel<<<1, 1>>>(buffer);
    cudaFree(buffer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 8);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 64);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu, 2097152);

// Benchmark current approach
template <int N>
__global__ void delete_kernel_2(ClassWithSize<N>* address) {
   delete address;
}

template <int N>
static void benchmark_new_on_gpu_and_copy_to_host(benchmark::State& state)
{
  while (state.KeepRunning()) {
    ClassWithSize<N>** gpuBuffer;
    cudaMalloc(&gpuBuffer, sizeof(ClassWithSize<N>*));
    create_kernel<<<1, 1>>>(gpuBuffer);
    ClassWithSize<N>** cpuBuffer = (ClassWithSize<N>**) malloc(sizeof(ClassWithSize<N>*));
    cudaMemcpy(cpuBuffer, gpuBuffer, sizeof(ClassWithSize<N>*), cudaMemcpyDeviceToHost);
    cudaFree(gpuBuffer);
    ClassWithSize<N>* gpuPointer = cpuBuffer[0];
    free(cpuBuffer);
    delete_kernel_2<<<1, 1>>>(gpuPointer);
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 8);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 64);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 512);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 4096);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 32768);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 262144);
BENCHMARK_TEMPLATE(benchmark_new_on_gpu_and_copy_to_host, 2097152);

// Benchmark how long it takes to create a stack object on the GPU
template <int N>
__global__ void create_on_stack_kernel() {
   (void) ClassWithSize<N>();
}

template <int N>
static void benchmark_create_on_stack_on_gpu(benchmark::State& state)
{
  while (state.KeepRunning()) {
    create_on_stack_kernel<N><<<1, 1>>>();
    cudaDeviceSynchronize();
  }
}

BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 8);
BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 64);
BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_create_on_stack_on_gpu, 2097152);

// Use managed_ptr
__global__ void fill(int numValues, int* values) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < numValues) {
      values[i] = i * i;
   }
}

__global__ void square(chai::managed_ptr<Base> object, int numValues, int* values) {
   object->scale(numValues, values);
}

void benchmark_use_managed_ptr_gpu(benchmark::State& state)
{
  chai::managed_ptr<Base> object = chai::make_managed<Derived>(2);

  int numValues = 100;
  int* values;
  cudaMalloc(&values, numValues * sizeof(int));
  fill<<<1, 100>>>(numValues, values);

  cudaDeviceSynchronize();

  while (state.KeepRunning()) {
    square<<<1, 1>>>(object, numValues, values);
    cudaDeviceSynchronize();
  }

  cudaFree(values);
  object.free();
  cudaDeviceSynchronize();
}

BENCHMARK(benchmark_use_managed_ptr_gpu);


// Curiously recurring template pattern
__global__ void square(BaseCRTP<DerivedCRTP> object, int numValues, int* values) {
   object.scale(numValues, values);
}

void benchmark_curiously_recurring_template_pattern_gpu(benchmark::State& state)
{
  BaseCRTP<DerivedCRTP>* derivedCRTP = new DerivedCRTP(2);
  auto object = *derivedCRTP;

  int numValues = 100;
  int* values;
  cudaMalloc(&values, numValues * sizeof(int));
  fill<<<1, 100>>>(numValues, values);

  cudaDeviceSynchronize();

  while (state.KeepRunning()) {
    square<<<1, 1>>>(object, numValues, values);
    cudaDeviceSynchronize();
  }

  cudaFree(values);
  delete derivedCRTP;
  cudaDeviceSynchronize();
}

BENCHMARK(benchmark_curiously_recurring_template_pattern_gpu);

// Class without inheritance
__global__ void square(NoInheritance object, int numValues, int* values) {
   object.scale(numValues, values);
}

void benchmark_no_inheritance_gpu(benchmark::State& state)
{
  NoInheritance* noInheritance = new NoInheritance(2);
  auto object = *noInheritance;

  int numValues = 100;
  int* values;
  cudaMalloc(&values, numValues * sizeof(int));
  fill<<<1, 100>>>(numValues, values);

  cudaDeviceSynchronize();

  while (state.KeepRunning()) {
    square<<<1, 1>>>(object, numValues, values);
    cudaDeviceSynchronize();
  }

  cudaFree(values);
  delete noInheritance;
  cudaDeviceSynchronize();
}

BENCHMARK(benchmark_no_inheritance_gpu);

__global__ void square(int numValues, int* values, chai::managed_ptr<Base> object) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < numValues) {
      int temp[4] = {i, i+1, i+2, i+3};
      object->sumAndScale(4, temp, values[i]);
   }
}

// managed_ptr (bulk)
template <int N>
void benchmark_bulk_use_managed_ptr_gpu(benchmark::State& state)
{
  chai::managed_ptr<Base> object = chai::make_managed<Derived>(2);

  int* values;
  cudaMalloc(&values, N * sizeof(int));
  fill<<<(N+255)/256, 256>>>(N, values);

  cudaDeviceSynchronize();

  while (state.KeepRunning()) {
    square<<<(N+255)/256, 256>>>(N, values, object);
    cudaDeviceSynchronize();
  }

  cudaFree(values);
  object.free();
  cudaDeviceSynchronize();
}

BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_gpu, 2097152);

// Curiously recurring template pattern
__global__ void square(int numValues, int* values, BaseCRTP<DerivedCRTP> object) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < numValues) {
      int temp[4] = {i, i+1, i+2, i+3};
      object.sumAndScale(4, temp, values[i]);
   }
}

template <int N>
void benchmark_bulk_curiously_recurring_template_pattern_gpu(benchmark::State& state)
{
  BaseCRTP<DerivedCRTP>* derivedCRTP = new DerivedCRTP(2);
  auto object = *derivedCRTP;

  int* values;
  cudaMalloc(&values, N * sizeof(int));
  fill<<<(N+255)/256, 256>>>(N, values);

  cudaDeviceSynchronize();

  while (state.KeepRunning()) {
    square<<<(N+255)/256, 256>>>(N, values, object);
    cudaDeviceSynchronize();
  }

  cudaFree(values);
  delete derivedCRTP;
  cudaDeviceSynchronize();
}

BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_gpu, 2097152);

// Class without inheritance
__global__ void square(int numValues, int* values, NoInheritance object) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < numValues) {
      int temp[4] = {i, i+1, i+2, i+3};
      object.sumAndScale(4, temp, values[i]);
   }
}

template <int N>
void benchmark_bulk_no_inheritance_gpu(benchmark::State& state)
{
  NoInheritance* noInheritance = new NoInheritance(2);
  auto object = *noInheritance;

  int* values;
  cudaMalloc(&values, N * sizeof(int));
  fill<<<(N+255)/256, 256>>>(N, values);

  cudaDeviceSynchronize();

  while (state.KeepRunning()) {
    square<<<(N+255)/256, 256>>>(N, values, object);
    cudaDeviceSynchronize();
  }

  cudaFree(values);
  delete noInheritance;
  cudaDeviceSynchronize();
}

BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_gpu, 2097152);

#endif

// managed_ptr
template <int N>
static void benchmark_bulk_polymorphism_cpu(benchmark::State& state)
{
  Base* object = new Derived(2);

  int* values = (int*) malloc(N * sizeof(int));

  for (int i = 0; i < N; ++i) {
     values[i] = i * i;
  }

#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif

  while (state.KeepRunning()) {
    for (int i = 0; i < N; ++i) {
       int temp[4] = {i, i+1, i+2, i+3};
       object->sumAndScale(4, temp, values[i]);
    }
  }

  delete object;
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
}

BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_polymorphism_cpu, 2097152);

// managed_ptr
template <int N>
static void benchmark_bulk_use_managed_ptr_cpu(benchmark::State& state)
{
  chai::managed_ptr<Base> object = chai::make_managed<Derived>(2);

  int* values = (int*) malloc(N * sizeof(int));

  for (int i = 0; i < N; ++i) {
     values[i] = i * i;
  }

#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif

  while (state.KeepRunning()) {
    for (int i = 0; i < N; ++i) {
       int temp[4] = {i, i+1, i+2, i+3};
       object->sumAndScale(4, temp, values[i]);
    }
  }

  object.free();
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
}

BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_use_managed_ptr_cpu, 2097152);

// Curiously recurring template pattern
template <int N>
static void benchmark_bulk_curiously_recurring_template_pattern_cpu(benchmark::State& state)
{
  BaseCRTP<DerivedCRTP>* object = new DerivedCRTP(2);

  int* values = (int*) malloc(N * sizeof(int));

  for (int i = 0; i < N; ++i) {
     values[i] = i * i;
  }

  while (state.KeepRunning()) {
    for (int i = 0; i < N; ++i) {
       int temp[4] = {i, i+1, i+2, i+3};
       object->sumAndScale(4, temp, values[i]);
    }
  }

  free(values);
  delete object;
}

BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_curiously_recurring_template_pattern_cpu, 2097152);

// Class without inheritance
template <int N>
static void benchmark_bulk_no_inheritance_cpu(benchmark::State& state)
{
  NoInheritance* object = new NoInheritance(2);

  int* values = (int*) malloc(N * sizeof(int));

  for (int i = 0; i < N; ++i) {
     values[i] = i * i;
  }

  while (state.KeepRunning()) {
    for (int i = 0; i < N; ++i) {
       int temp[4] = {i, i+1, i+2, i+3};
       object->sumAndScale(4, temp, values[i]);
    }
  }

  free(values);
  delete object;
}

BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 1);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 256);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 512);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 1024);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 2048);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 4096);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 8192);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 16384);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 32768);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 65536);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 131072);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 262144);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 524288);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 1048576);
BENCHMARK_TEMPLATE(benchmark_bulk_no_inheritance_cpu, 2097152);

BENCHMARK_MAIN();

