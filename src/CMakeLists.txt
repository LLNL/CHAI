set (chai_headers
  ArrayManager.hpp
  ArrayManager.inl
  ChaiMacros.hpp
  ExecutionSpaces.hpp
  ManagedArray.hpp
  ManagedArray.inl
  PointerRecord.hpp
  Types.hpp)

set (chai_sources
  ArrayManager.cpp)

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  DEPENDS_ON cuda)

target_include_directories(
  chai
  PRIVATE ${PROJECT_BINARY_DIR}/include)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/chai-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake
  DESTINATION share/chai/cmake/)

install(EXPORT chai-targets DESTINATION share/chai/cmake/)

add_subdirectory(tests)
if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_subdirectory(examples)
