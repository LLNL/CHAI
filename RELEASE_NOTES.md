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

## [Unreleased] - Release date yyyy-mm-dd

### Fixed
- Fixes reallocate when using pinned or unified memory.
- Adds missing synchronize when using pinned memory.
- Fixes possible hangs when evicting data.
- Now respects allocators passed to ManagedArray constructors when CHAI\_DISABLE\_RM=TRUE.

### Removed
- Removes deprecated ManagedArray::getPointer method. Use ManagedArray::data instead.
- Removes ManagedArray::incr and ManagedArray::decr methods. Use ManagedArray::pick and ManagedArray::set instead.
- Removes optional support for implicitly casting between raw pointers and ManagedArrays (CHAI\_ENABLE\_IMPLICIT\_CONVERSIONS). Use makeManagedArray and ManagedArray::data to perform explicit conversions instead.
- Removes equality and inequality comparison operators between ManagedArrays and raw pointers.
- Removes make\_managed\_from\_factory function for creating managed\_ptr objects from factory functions. This change will lead to safer adoption of allocators during construction and destruction of managed\_ptr objects.
- Removes CHAI\_ENABLE\_PICK CMake option. ManagedArray::pick and ManagedArray::set will always be available.

## [Version 2024.07.0] - Release date 2024-07-26

### Changed
- Updated to Umpire v2024.07.0 and RAJA v2024.07.0
- Increased minimum CMake version required to 3.23
- Turns off benchmarks and examples by default
- Enable RAJA plugin by default

## [Version 2024.02.2] - Release date 2024-06-26

### Changed
- Updated to RAJA v2024.02.2
- Improved debugging with GPU simulation mode

## [Version 2024.02.1] - Release date 2024-04-19

### Changed
- Updated to BLT v0.6.2, Umpire v2024.02.1, and RAJA v2024.02.1

## [Version 2024.02.0] - Release date 2024-03-04

### Added
- Support for APUs with a single memory space. To use, configure with -DCHAI\_DISABLE\_RM=ON -DCHAI\_THIN\_GPU\_ALLOCATE=ON.

### Changed
- Moved installed CMake targets from share/chai/cmake to lib/cmake/chai to be consistent with other libraries in the RAJA Portability Suite
- Improved dependency handling during the build of CHAI and when it is imported into another library/application
- Removed ArrayManager::enableDeviceSynchronization and ArrayManager::disableDeviceSynchronization. Instead, use the environment variables for device synchronization after all kernels (e.g. CUDA\_LAUNCH\_BLOCKING or HIP\_LAUNCH\_BLOCKING)

### Fixed
- Use free instead of realloc when the size is 0 (fixes a warning from valgrind)
