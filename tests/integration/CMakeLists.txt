# Integration test dependencies
set (chai_integration_test_depends
     chai umpire gtest)

blt_list_append(TO chai_integration_test_depends ELEMENTS cuda IF ${ENABLE_CUDA})
blt_list_append(TO chai_integration_test_depends ELEMENTS hip IF ${ENABLE_HIP})

# ManagedArray tests
blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON ${chai_integration_test_depends})

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)

# managed_ptr tests
blt_add_executable(
  NAME managed_ptr_tests
  SOURCES managed_ptr_tests.cpp
  DEPENDS_ON ${chai_integration_test_depends})

target_include_directories(
  managed_ptr_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_ptr_test
  COMMAND managed_ptr_tests)

