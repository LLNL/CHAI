##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set(ENABLE_FORTRAN Off CACHE BOOL "Enable Fortran in Umpire")

if (NOT TARGET umpire)
  if (DEFINED umpire_DIR)
    find_package(umpire REQUIRED)

    blt_register_library(
      NAME umpire
      INCLUDES ${UMPIRE_INCLUDE_DIRS}
      LIBRARIES umpire)
  else ()
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/umpire)
  endif()
endif()

blt_register_library(
  NAME umpire
  INCLUDES ${UMPIRE_INCLUDE_DIRS}
  LIBRARIES umpire)

if (ENABLE_RAJA_PLUGIN)
  find_package(camp REQUIRED)
  find_package(RAJA REQUIRED)
  
  blt_register_library(
    NAME raja
    INCLUDES ${RAJA_INCLUDE_DIR}
    LIBRARIES RAJA)

  message(STATUS "RAJA: ${RAJA_INCLUDE_DIR}")
endif ()
