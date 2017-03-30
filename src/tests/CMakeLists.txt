blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON chai gtest cuda)

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)
