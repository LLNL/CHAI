##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_umpire_example_depends
  chai)

if (ENABLE_CUDA)
  set (chai_umpire_example_depends
    ${chai_umpire_example_depends}
    cuda)
endif()
if (ENABLE_HIP)
  set (chai_umpire_example_depends
    ${chai_umpire_example_depends}
    hip)
endif()

blt_add_executable(
  NAME chai-umpire-example.exe
  SOURCES chai-umpire-allocators.cpp
  DEPENDS_ON ${chai_umpire_example_depends})

if (ENABLE_CUDA OR ENABLE_HIP)
  blt_add_executable(
    NAME chai-example.exe
    SOURCES example.cpp
    DEPENDS_ON ${chai_umpire_example_depends})

  blt_add_executable(
    NAME pinned.exe
    SOURCES pinned.cpp
    DEPENDS_ON ${chai_umpire_example_depends})
endif ()

if (CHAI_ENABLE_MANAGED_PTR)
  blt_add_executable(
    NAME managed_ptr_example.exe
    SOURCES managed_ptr_example.cpp
    DEPENDS_ON ${chai_umpire_example_depends})
endif ()

