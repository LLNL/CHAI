[comment]: # (#################################################################)
[comment]: # (Copyright 2016-25, Lawrence Livermore National Security, LLC)
[comment]: # (and CHAI project contributors. See the CHAI LICENSE file for)
[comment]: # (details.)
[comment]: # 
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# CHAI

[![Azure Build Status](https://dev.azure.com/davidbeckingsale/CHAI/_apis/build/status/LLNL.CHAI?branchName=develop)](https://dev.azure.com/davidbeckingsale/CHAI/_build/latest?definitionId=2&branchName=develop)
[![Build Status](https://travis-ci.org/LLNL/CHAI.svg?branch=develop)](https://travis-ci.org/LLNL/CHAI)
[![Documentation Status](https://readthedocs.org/projects/chai/badge/?version=develop)](https://chai.readthedocs.io/en/develop/?badge=develop)


CHAI is a library that handles automatic data migration to different memory
spaces behind an array-style interface. It was designed to work with
[RAJA](https://github.com/LLNL/RAJA) and integrates with it. CHAI could be
used with other C++ abstractions, as well.

CHAI uses CMake and BLT to handle builds. Make sure that you have a modern
compiler loaded and the configuration is as simple as:

    $ git submodule update --init --recursive
    $ mkdir build && cd build
    $ cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ../

CMake will provide output about which compiler is being used, and what version
of CUDA was detected. Once CMake has completed, CHAI can be built with Make:

    $ make

For more advanced configuration you can use standard CMake variables.

More information is available in the [CHAI documentation](https://chai.readthedocs.io/en/develop/).

## Authors

The original developers of CHAI are:

- Holger Jones (jones19@llnl.gov)
- David Poliakoff (poliakoff1@llnl.gov)
- Peter Robinson (robinson96@llnl.gov)

Contributors include:

- David Beckingsale (david@llnl.gov)
- Riyaz Haque (haque1@llnl.gov)
- Adam Kunen (kunen1@llnl.gov)

## Release

Copyright (c) 2016, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory

All rights reserved.

Unlimited Open Source - BSD Distribution

For release details and restrictions, please read the LICENSE file.
It is also linked here: [LICENSE](./LICENSE)

- `LLNL-CODE-705877`
- `OCEC-16-189`
