set (managed_array_depends
  chai gtest)

if (ENABLE_CUDA)
  set (managed_array_test_depends
    ${managed_array_depends}
    cuda)
endif ()

blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON ${managad_array_test_depends})

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)
