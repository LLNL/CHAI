##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_reproducer_depends chai)

if (CHAI_ENABLE_CUDA)
  list(APPEND chai_reproducer_depends cuda)
endif ()

if (CHAI_ENABLE_HIP)
  list(APPEND chai_reproducer_depends blt::hip)
endif ()

if (CHAI_ENABLE_MANAGED_PTR AND (CHAI_ENABLE_HIP OR CHAI_ENABLE_CUDA))
  # This crashes with HIP but not with CUDA
  blt_add_executable(
    NAME managed_ptr_multiple_inheritance_reproducer.exe
    SOURCES managed_ptr_multiple_inheritance_reproducer.cpp
    DEPENDS_ON ${chai_reproducer_depends})
endif ()

if (CHAI_ENABLE_MANAGED_PTR AND CHAI_ENABLE_HIP)
  blt_add_executable(
    NAME virtual_function_simple_reproducer.exe
    SOURCES virtual_function_simple_reproducer.cpp
    DEPENDS_ON ${chai_reproducer_depends})

  blt_add_executable(
    NAME virtual_function_complex_reproducer.exe
    SOURCES virtual_function_complex_reproducer.cpp
    DEPENDS_ON ${chai_reproducer_depends})

  blt_add_executable(
    NAME managed_ptr_reproducer.exe
    SOURCES managed_ptr_reproducer.cpp
    DEPENDS_ON ${chai_reproducer_depends})
endif ()

