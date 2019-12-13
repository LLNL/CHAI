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
#include "chai/ManagedArray.hpp"
#include "chai/ManagedArrayView.hpp"

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

int main(int, char **)
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

  std::cout << "   _____ _    _          _____  " << std::endl;
  std::cout << "  / ____| |  | |   /\\   |_   _| " << std::endl;
  std::cout << "  | |    | |__| |  /  \\    | |  " << std::endl;
  std::cout << "  | |    |  __  | / /\\ \\   | |  " << std::endl;
  std::cout << "  | |____| |  | |/ ____ \\ _| |_ " << std::endl;
  std::cout << "  \\_____|_|  |_/_/    \\_\\_____| " << std::endl;
  std::cout << "\n\n" << std::endl;

  auto I_array = chai::ManagedArray<double>(NN);
  auto Iold_array = chai::ManagedArray<double>(NN);

  chai::ManagedArrayView<double, RAJA::Layout<2>> I(I_array, N+2, N+2);
  chai::ManagedArrayView<double, RAJA::Layout<2>> Iold(Iold_array, N+2, N+2);

  RAJA::RangeSegment gridRange(0, NN);
  RAJA::RangeSegment jacobiRange(1, (N + 1));

  RAJA::forall<RAJA::seq_exec>(gridRange, [=] (int i) {
      I_array[i] = 0.0;
      Iold_array[i] = 0.0;
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

          I(m,n) = 0.25 * (-f + Iold(m,n-1) + Iold(m,n+1) 
              + Iold(m-1,n) + Iold(m+1,n));
    });

    RAJA::ReduceSum<RAJA::seq_reduce, double> RAJA_resI2(0.0);
    RAJA::forall<RAJA::seq_exec>(
      gridRange, [=](RAJA::Index_type k) {
        RAJA_resI2 += (I_array[k] - Iold_array[k]) * (I_array[k] - Iold_array[k]);
        Iold_array[k] = I_array[k];
      });
    
    resI2 = RAJA_resI2;

    if (iteration > maxIter) {
      printf("Jacobi: Sequential - Maxed out on iterations! \n");
      exit(-1);
    }

    iteration++;
  }

  computeErr(I_array, gridx);
  printf("No of iterations: %d \n \n", iteration);
  
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
