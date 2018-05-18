###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

set (raja_test_depends
  chai umpire raja openmp gtest)

blt_add_executable(
  NAME raja-chai-tests 
  SOURCES raja-chai-tests.cpp
  DEPENDS_ON ${raja_test_depends})

blt_add_test(
  NAME raja-chai-tests
  COMMAND raja-chai-tests)

target_include_directories(
  raja-chai-tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)
