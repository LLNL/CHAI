##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

FROM ghcr.io/llnl/radiuss:ubuntu-24.04-gcc-13 AS gcc
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:ubuntu-24.04-clang-19 AS clang
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:ubuntu-24.04-intel-2024.2 AS intel
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16 && \
    ctest -T test --output-on-failure"

FROM ghcr.io/llnl/radiuss:ubuntu-24.04-cuda-12-9 AS cuda
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DENABLE_CUDA=On -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=70 .. && \
    make -j 16

FROM ghcr.io/llnl/radiuss:ubuntu-24.04-hip-6.4.3 AS hip
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DENABLE_HIP=On -DENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16

FROM ghcr.io/llnl/radiuss:ubuntu-24.04-intel-2024.2 AS sycl
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_C_COMPILER=icx -DENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16"
