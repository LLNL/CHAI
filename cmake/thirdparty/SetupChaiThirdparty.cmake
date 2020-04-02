##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
if (NOT TARGET umpire)
  if (DEFINED umpire_DIR)
    # this allows umpire_DIR to be the install prefix if we are relying on umpire's
    # installed umpire-config.cmake
    list(APPEND CMAKE_PREFIX_PATH ${umpire_DIR})
    find_package(umpire REQUIRED)
    if (ENABLE_MPI)
      set(UMPIRE_DEPENDS mpi)
    else()
      set(UMPIRE_DEPENDS)
    endif()
    blt_register_library(
      NAME umpire
      INCLUDES ${UMPIRE_INCLUDE_DIRS}
      LIBRARIES umpire
      DEPENDS_ON ${UMPIRE_DEPENDS})
  else ()
    set(OLD_ENABLE_FORTRAN ${ENABLE_FORTRAN})
    set(ENABLE_FORTRAN Off CACHE BOOL "Enable Fortran in Umpire")
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/umpire)
    set(ENABLE_FORTRAN ${OLD_ENABLE_FORTRAN})
  endif()
endif()

if (ENABLE_RAJA_PLUGIN)
  if (NOT TARGET RAJA)
    if (DEFINED RAJA_DIR)
      # this allows RAJA_DIR to be the install prefix if we are relying on RAJA's
      # installed RAJA-config.cmake
      list(APPEND CMAKE_PREFIX_PATH ${RAJA_DIR})
      message(STATUS "CHAI: using external RAJA via find_package ${RAJA_DIR}")
      find_package(RAJA REQUIRED)
       blt_register_library(
         NAME RAJA
         INCLUDES ${RAJA_INCLUDE_DIRS}
         LIBRARIES RAJA)
    else()
      message(STATUS "CHAI: using builtin RAJA submodule")
      add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/raja)
    endif()
  else()
    message(STATUS "CHAI: using existing RAJA target")
  endif()
endif ()
