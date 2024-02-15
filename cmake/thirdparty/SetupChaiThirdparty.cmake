##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

if (NOT TARGET umpire)
  find_package(umpire CONFIG QUIET NO_DEFAULT_PATH PATHS ${umpire_DIR})

  if (NOT umpire_FOUND)
    if (NOT EXISTS ${PROJECT_SOURCE_DIR}/src/tpl/umpire/CMakeLists.txt)
      message(FATAL_ERROR "[CHAI] Umpire not found! Set umpire_DIR to the install location of Umpire or run 'git submodule update --init --recursive' in the CHAI repository, then try building again.")
    else ()
      set(UMPIRE_ENABLE_FORTRAN Off CACHE BOOL "Enable Fortran in Umpire")
      set(UMPIRE_ENABLE_C Off CACHE BOOL "Enable Fortran in Umpire")
      set(UMPIRE_ENABLE_TESTS Off CACHE BOOL "")
      set(UMPIRE_ENABLE_TOOLS Off CACHE BOOL "")
      add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/umpire)
    endif ()
  endif ()
endif ()

if (CHAI_ENABLE_RAJA_PLUGIN)
  if (NOT TARGET RAJA)
    find_package(raja CONFIG QUIET NO_DEFAULT_PATH PATHS ${raja_DIR})

    if (NOT raja_FOUND)
      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/src/tpl/raja/CMakeLists.txt)
        message(FATAL_ERROR "[CHAI] RAJA not found! Set raja_DIR to the install location of RAJA or run 'git submodule update --init --recursive' in the CHAI repository, then try building again.")
      else ()
        add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/raja)
      endif ()
    endif ()
  endif ()
endif ()
