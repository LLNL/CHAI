set (managed_array_test_depends
  chai umpire gtest)

if (ENABLE_CUDA)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    cuda)
endif ()

if (ENABLE_HIP)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    hip)
endif ()

if (ENABLE_OPENMP)
  set (managed_array_test_depends
    ${managed_array_test_depends}
    openmp)
endif ()

blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON ${managed_array_test_depends})

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)

set(raja_test_depends
  ${managed_array_test_depends}
  raja)

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
