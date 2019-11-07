##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
if (NOT TARGET umpire)
  if (DEFINED umpire_DIR)
    find_package(umpire REQUIRED)

    blt_register_library(
      NAME umpire
      INCLUDES ${UMPIRE_INCLUDE_DIRS}
      LIBRARIES umpire)
  else ()
    set(OLD_ENABLE_FORTRAN ${ENABLE_FORTRAN})
    set(ENABLE_FORTRAN Off CACHE BOOL "Enable Fortran in Umpire")
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/umpire)
    set(ENABLE_FORTRAN ${OLD_ENABLE_FORTRAN})
  endif()
endif()

if (ENABLE_RAJA_PLUGIN)
  if (NOT TARGET RAJA)
    if (DEFINED raja_DIR)
      message(STATUS "CHAI: using external RAJA via find_package")
      find_package(RAJA REQUIRED)
    else()
      message(STATUS "CHAI: using builtin RAJA submodule")
      add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/raja)
    endif()
  else()
    message(STATUS "CHAI: using existing RAJA target")
  endif()
endif ()
