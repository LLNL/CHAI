##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
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
    set(UMPIRE_ENABLE_FORTRAN Off CACHE BOOL "Enable Fortran in Umpire")
    set(UMPIRE_ENABLE_C Off CACHE BOOL "Enable Fortran in Umpire")
    set(UMPIRE_ENABLE_TESTS Off CACHE BOOL "")
    set(UMPIRE_ENABLE_TOOLS Off CACHE BOOL "")
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/umpire)
  endif()

  # Umpire depends on camp
  if (NOT TARGET camp)
    if (DEFINED camp_DIR)
      find_package(camp REQUIRED)
      set_target_properties(camp PROPERTIES IMPORTED_GLOBAL TRUE)
    else ()
      message(FATAL_ERROR "camp is required. Please set camp_DIR")
    endif()

    if(ENABLE_CUDA)
      blt_add_target_definitions(
        TO camp
        SCOPE INTERFACE
        TARGET_DEFINITIONS CAMP_HAVE_CUDA)
    endif()
  endif ()
endif()

if (CHAI_ENABLE_RAJA_PLUGIN)
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
