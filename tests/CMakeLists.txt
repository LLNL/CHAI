##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

add_subdirectory(install)
add_subdirectory(unit)
add_subdirectory(integration)

if(CHAI_ENABLE_EXPERIMENTAL)
  add_subdirectory(expt)
endif()
