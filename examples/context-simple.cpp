#include "camp/device.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

int main()
{
  camp::devices::Context host{camp::devices::Host{}}; 
  camp::devices::Context device{camp::devices::Cuda{}}; 

  constexpr std::size_t ARRAY_SIZE{1024};

  chai::ManagedArray<double> array(ARRAY_SIZE);

  // set on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = i;
  }); 


  // double on device
  forall(&device, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array[i] = array[i] * 2.0;
  });

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      printf("array[%d] = %f \n", i, array[i]);
  }); 
}
