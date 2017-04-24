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
