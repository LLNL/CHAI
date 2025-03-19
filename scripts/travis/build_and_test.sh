#!/bin/bash
##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

threads=3
top_dir=$(pwd)
travis_build_dir=${top_dir}/travis-build
travis_install_dir=${top_dir}/travis-install

or_die mkdir $travis_build_dir
or_die mkdir $travis_install_dir

if [[ "$DO_BUILD" == "yes" ]] ; then
    or_die cd $travis_build_dir
    or_die cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} -DCMAKE_INSTALL_PREFIX=$travis_install_dir ../
    or_die make -j $threads VERBOSE=1
    or_die make install
    if [[ "${DO_TEST}" == "yes" ]] ; then
        or_die ctest -V
    fi
fi

if [[ "$BUILD_RAJA" == "yes" ]] ; then
  or_die cd $top_dir
  or_die git clone --recursive -b develop https://github.com/LLNL/RAJA.git
  or_die cd RAJA
  or_die mkdir build
  or_die cd build
  or_die cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} -DENABLE_CHAI=On -Dchai_DIR=${travis_install_dir}/share/chai/cmake ../
  or_die make -j $threads VERBOSE=2
fi

exit 0
