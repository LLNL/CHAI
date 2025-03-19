##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

if (NOT TARGET umpire)
  if (DEFINED umpire_DIR OR DEFINED UMPIRE_DIR)
    message(STATUS "[CHAI] Using external Umpire")
    find_package(umpire CONFIG REQUIRED NO_DEFAULT_PATH PATHS ${umpire_DIR} ${UMPIRE_DIR})
  else ()
    if (NOT EXISTS ${PROJECT_SOURCE_DIR}/src/tpl/umpire/CMakeLists.txt)
      message(FATAL_ERROR "[CHAI] Umpire not found! Set umpire_DIR to the install location of Umpire or run 'git submodule update --init --recursive' in the CHAI repository, then try building again.")
    else ()
      message(STATUS "[CHAI] Using internal Umpire")

      set(UMPIRE_ENABLE_BENCHMARKS Off CACHE BOOL "Enable benchmarks in Umpire")
      set(UMPIRE_ENABLE_C Off CACHE BOOL "Enable C in Umpire")
      set(UMPIRE_ENABLE_DOCS Off CACHE BOOL "Enable documentation in Umpire")
      set(UMPIRE_ENABLE_EXAMPLES Off CACHE BOOL "Enable examples in Umpire")
      set(UMPIRE_ENABLE_FORTRAN Off CACHE BOOL "Enable Fortran in Umpire")
      set(UMPIRE_ENABLE_TESTS Off CACHE BOOL "Enable tests in Umpire")
      set(UMPIRE_ENABLE_TOOLS Off CACHE BOOL "Enable tools in Umpire")

      add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/umpire)
    endif ()
  endif ()
endif ()

if (CHAI_ENABLE_RAJA_PLUGIN)
  if (NOT TARGET RAJA)
    if (DEFINED raja_DIR OR DEFINED RAJA_DIR)
      message(STATUS "[CHAI] Using external RAJA")
      find_package(raja CONFIG REQUIRED NO_DEFAULT_PATH PATHS ${raja_DIR} ${RAJA_DIR})
    else ()
      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/src/tpl/raja/CMakeLists.txt)
        message(FATAL_ERROR "[CHAI] RAJA not found! Set raja_DIR to the install location of RAJA or run 'git submodule update --init --recursive' in the CHAI repository, then try building again.")
      else ()
        message(STATUS "[CHAI] Using internal RAJA")
        set(RAJA_ENABLE_BENCHMARKS Off CACHE BOOL "Enable benchmarks in RAJA")
        set(RAJA_ENABLE_DOCS Off CACHE BOOL "Enable documentation in RAJA")
        set(RAJA_ENABLE_EXAMPLES Off CACHE BOOL "Enable examples in RAJA")
        set(RAJA_ENABLE_EXERCISES Off CACHE BOOL "Enable exercises in RAJA")
        set(RAJA_ENABLE_TESTS Off CACHE BOOL "Enable tests in RAJA")
        add_subdirectory(${PROJECT_SOURCE_DIR}/src/tpl/raja)
      endif ()
    endif ()
  endif ()
endif ()
