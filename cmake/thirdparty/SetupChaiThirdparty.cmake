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

if (NOT TARGET camp)
  if (DEFINED camp_DIR)
    find_package(camp REQUIRED)

    blt_register_library(
      NAME camp
      INCLUDES ${CAMP_INCLUDE_DIRS}
      LIBRARIES camp)
  else ()
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/camp)
  endif()
endif()
