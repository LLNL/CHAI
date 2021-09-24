##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_reproducer_depends chai)

if (ENABLE_CUDA)
  list(APPEND chai_reproducer_depends cuda)
endif ()

if (ENABLE_HIP)
  list(APPEND chai_reproducer_depends hip)
endif ()

if (CHAI_ENABLE_MANAGED_PTR AND ENABLE_HIP)
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

