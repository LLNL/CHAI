//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <iostream>
#include <cmath>

#include "RAJA/RAJA.hpp"

//
//  Struct to hold grid info
//  o - Origin in a cartesian dimension
//  h - Spacing between grid points
//  n - Number of grid points
//
struct grid_s {
  double o, h;
  int n;
};

double solution(double x, double y);
void computeErr(double *I, grid_s grid);

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  double tol = 1e-10;

  int N = 50;
  int NN = (N + 2) * (N + 2);
  int maxIter = 100000;

  double resI2;
  int iteration;

  grid_s gridx;
  gridx.o = 0.0;
  gridx.h = 1.0 / (N + 1.0);
  gridx.n = N + 2;

  auto I = chai::ManagedArray<double>(NN);
  auto Iold = chai::ManagedArray<double>(NN);

  RAJA::RangeSegment gridRange(0, NN);
  RAJA::RangeSegment jacobiRange(1, (N + 1));

  RAJA::forall<RAJA::seq_exec>(gridRange, [=] (int i) {
      I[i] = 0.0;
      Iold[i] = 0.0;
  });

  using jacobiSeqNestedPolicy = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec, 
          RAJA::statement::Lambda<0>
        > 
      >
    >;

  resI2 = 1;
  iteration = 0;

  while (resI2 > tol * tol) {

    RAJA::kernel<jacobiSeqNestedPolicy>(RAJA::make_tuple(jacobiRange,jacobiRange),
                         [=] (RAJA::Index_type m, RAJA::Index_type n) {
          double x = gridx.o + m * gridx.h;
          double y = gridx.o + n * gridx.h;

          double f = gridx.h * gridx.h
                     * (2 * x * (y - 1) * (y - 2 * x + x * y + 2) * exp(x - y));

          int id = n * (N + 2) + m;
          I[id] =
               0.25 * (-f + Iold[id - N - 2] + Iold[id + N + 2] + Iold[id - 1]
                          + Iold[id + 1]);
    });

    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_exec>(
      gridRange, [=](RAJA::Index_type k) {
        RAJA_resI2 += (I[k] - Iold[k]) * (I[k] - Iold[k]);          
        Iold[k] = I[k];
      });
    
    resI2 = RAJA_resI2;

    if (iteration > maxIter) {
      printf("Jacobi: Sequential - Maxed out on iterations! \n");
      exit(-1);
    }

    iteration++;
  }

  computeErr(I, gridx);
  printf("No of iterations: %d \n \n", iteration);
  
  memoryManager::deallocate(I);
  memoryManager::deallocate(Iold);
  
  return 0;
}

//
// Function for the anlytic solution
//
double solution(double x, double y)
{
  return x * y * exp(x - y) * (1 - x) * (1 - y);
}

//
// Error is computed via ||I_{approx}(:) - U_{analytic}(:)||_{inf}
//
void computeErr(double *I, grid_s grid)
{

  RAJA::RangeSegment gridRange(0, grid.n);
  RAJA::ReduceMax<RAJA::seq_reduce, double> tMax(-1.0);

  using jacobiSeqNestedPolicy = RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,
      RAJA::statement::For<0, RAJA::seq_exec, RAJA::statement::Lambda<0> > > >;

  RAJA::kernel<jacobiSeqNestedPolicy>(RAJA::make_tuple(gridRange,gridRange),
                       [=] (RAJA::Index_type ty, RAJA::Index_type tx ) {

      int id = tx + grid.n * ty;
      double x = grid.o + tx * grid.h;
      double y = grid.o + ty * grid.h;
      double myErr = std::abs(I[id] - solution(x, y));
      tMax.max(myErr);
    });

  double l2err = tMax;
  printf("Max error = %lg, h = %f \n", l2err, grid.h);
}
