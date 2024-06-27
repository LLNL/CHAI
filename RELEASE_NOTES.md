[comment]: # (#################################################################)
[comment]: # (Copyright 2016-24, Lawrence Livermore National Security, LLC)
[comment]: # (and CHAI project contributors. See the CHAI LICENSE file for)
[comment]: # (details.)
[comment]: # 
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# CHAI Software Release Notes

Notes describing significant changes in each CHAI release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Version 2024.07.0] - Release date 2024-07

### Changed
- Updated to Umpire v2024.07.0 and RAJA v2024.07.0
- Increased minimum CMake version required to 3.20
- Turns off benchmarks and examples by default

## [Version 2024.02.2] - Release date 2024-06-26

### Changed
- Updated to RAJA v2024.02.2
- Improved debugging with GPU simulation mode

## [Version 2024.02.1] - Release date 2024-04-19

### Changed
- Updated to BLT v0.6.2, Umpire v2024.02.1, and RAJA v2024.02.1

## [Version 2024.02.0] - Release date 2024-03-04

### Added
- Support for APUs with a single memory space. To use, configure with -DCHAI\_DISABLE\_RM=ON -DCHAI\_GPU\_THIN\_ALLOCATE=ON.

### Changed
- Moved installed CMake targets from share/chai/cmake to lib/cmake/chai to be consistent with other libraries in the RAJA Portability Suite
- Improved dependency handling during the build of CHAI and when it is imported into another library/application
- Removed ArrayManager::enableDeviceSynchronization and ArrayManager::disableDeviceSynchronization. Instead, use the environment variables for device synchronization after all kernels (e.g. CUDA\_LAUNCH\_BLOCKING or HIP\_LAUNCH\_BLOCKING)

### Fixed
- Use free instead of realloc when the size is 0 (fixes a warning from valgrind)
