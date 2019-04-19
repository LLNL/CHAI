FROM axom/compilers:gcc-5 AS gcc5
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test

FROM axom/compilers:gcc-6 AS gcc6
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test

FROM axom/compilers:gcc-7 AS gcc7
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test

FROM axom/compilers:gcc-8 AS gcc8
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test

FROM axom/compilers:clang-4 AS clang4
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test

FROM axom/compilers:clang-5 AS clang5
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test

FROM axom/compilers:clang-6 AS clang6
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_CUDA=OFF ..
RUN cd build && make -j 16
RUN cd build && ctest -T test


FROM axom/compilers:nvcc-9 AS nvcc
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On ..
RUN cd build && make -j 16
