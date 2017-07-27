# CHAI

CHAI is a libary handling automatic data migration to different memory spaces
behind an array-style interface.

CHAI uses CMake and BLT to handle builds. Make sure that you have a modern
compiler loaded and the configuration is as simple as:

    $ mkdir build && cd build
    $ cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ../

CMake will provide output about which compiler is being used, and what version
of CUDA was detected. Once CMake has completed, CHAI can be built with Make:

    $ make

For more advanced configuration you can use standard CMake variables.

More information is available in the CHAI documentation.

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
