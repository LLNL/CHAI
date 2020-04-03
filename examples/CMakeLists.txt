##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
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

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)

if (ENABLE_CUDA OR ENABLE_HIP)
  blt_add_executable(
    NAME chai-example.exe
    SOURCES example.cpp
    DEPENDS_ON ${chai_umpire_example_depends})
endif ()

