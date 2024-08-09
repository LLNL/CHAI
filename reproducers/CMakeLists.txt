##############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_reproducer_depends chai)

blt_add_executable(
  NAME c_array_of_managed_arrays_reproducer
  SOURCES c_array_of_managed_arrays_reproducer.cpp
  DEPENDS_ON ${chai_reproducer_depends})

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

