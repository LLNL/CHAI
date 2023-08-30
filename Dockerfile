FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-7.3.0 AS gcc7
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-8.1.0 AS gcc8
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-18.04:gcc-9.4.0 AS gcc9
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/gcc-ubuntu-18.04:gcc-11.2.0 AS gcc11
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-10.0.0 AS clang10
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-11.0.0 AS clang11
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-12.0.0 AS clang12
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-13.0.0 AS clang13
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j 16 && \
    ctest -T test --output-on-failure

FROM ghcr.io/rse-ops/cuda:cuda-10.1.243-ubuntu-18.04 AS nvcc10
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc11
ENV GTEST_COLOR=1
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
    cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On .. && \
    make -j 16

FROM ghcr.io/rse-ops/hip-ubuntu-22.04:hip-4.3.1 AS hip
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/chai/workspace
WORKDIR /home/chai/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && spack load hip llvm-amdgpu && \
    cmake -DBLT_EXPORT_THIRDPARTY=On -DENABLE_WARNINGS_AS_ERRORS=Off -DCHAI_ENABLE_MANAGED_PTR=Off -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DENABLE_HIP=On .. && \
    make -j 16 VERBOSE=1