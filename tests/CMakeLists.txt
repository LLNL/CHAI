##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

add_subdirectory(install)
add_subdirectory(unit)
add_subdirectory(integration)

if(CHAI_ENABLE_EXPERIMENTAL)
  add_subdirectory(expt)
endif()
