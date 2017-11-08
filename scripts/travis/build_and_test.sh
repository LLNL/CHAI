#!/bin/bash

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

source ~/.bashrc
cd ${TRAVIS_BUILD_DIR}
or_die mkdir travis-build
or_die mkdir travis-install

cd travis-build

if [[ "$DO_BUILD" == "yes" ]] ; then
    or_die cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} -DCMAKE_INSTALL_PREFIX=../travis-install/ ../
    or_die make -j 3 VERBOSE=1
    if [[ "${DO_TEST}" == "yes" ]] ; then
        or_die ctest -V
    fi
fi

if [[ "$BUILD_RAJA" == "yes" ]] ; then
  or_die cd ${TRAVIS_BUILD_DIR}/../
  or_die git clone --recursive -b develop https://github.com/LLNL/RAJA.git
  or_die cd RAJA
  or_die mkdir build
  or_die cd build
  or_die cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} -DENABLE_CHAI=On -Dchai_DIR=${TRAVIS_BUILD_DIR}/travis-install ../
  or_die make -j 3 VERBOSE=2
fi

exit 0
