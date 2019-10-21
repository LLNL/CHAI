#include "camp/resources.hpp"
#include "../src/util/forall.hpp"
#include "chai/ManagedArray.hpp"

int main()
{
  camp::resources::Context host{camp::resources::Host{}}; 

  camp::resources::Context device_one{camp::resources::Cuda{}}; 
  camp::resources::Context device_two{camp::resources::Cuda{}}; 

  constexpr std::size_t ARRAY_SIZE{1024};

  chai::ManagedArray<double> array_one(ARRAY_SIZE);
  chai::ManagedArray<double> array_two(ARRAY_SIZE);

  // set on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array_one[i] = i;
      array_two[i] = i;
  }); 


  // double on device
  forall(&device_one, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array_one[i] = array_one[i] * 2.0;
  });
  forall(&device_two, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      array_two[i] = array_two[i] / 2.0;
  });

  array_one.move(chai::CPU, &device_one);
  array_two.move(chai::CPU, &device_two);

  // print on host
  forall(&host, 0, ARRAY_SIZE, [=] __host__ __device__ (int i) {
      if (i == 256) {
        printf("array_one[%d] = %f \n", i, array_one[i]);
        printf("array_two[%d] = %f \n", i, array_two[i]);
      }
  }); 
}
