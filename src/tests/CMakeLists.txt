blt_add_executable(
  NAME managed_array_tests
  SOURCES managed_array_tests.cpp
  DEPENDS_ON gtest chai
  USE_CUDA True)

target_include_directories(
  managed_array_tests
  PUBLIC ${PROJECT_SOURCE_DIR}/src/components/chai)

blt_add_test(
  NAME managed_array_test
  COMMAND managed_array_tests)
