configure_file(
  ${PROJECT_SOURCE_DIR}/src/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/chai/config.hpp)

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

set (chai_depends
  cuda)

if (ENABLE_CNMEM)
  set (chai_depends
    ${chai_depends}
    cnmem)
endif()

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  DEPENDS_ON ${chai_depends})

target_include_directories(
  chai
  PRIVATE ${PROJECT_BINARY_DIR}/include)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/chai-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/chai/cmake/chai-config.cmake
  DESTINATION share/chai/cmake/)

install(
  FILES ${PROJECT_BINARY_DIR}/include/chai/config.hpp
  DESTINATION include/chai)

install(EXPORT chai-targets DESTINATION share/chai/cmake/)

add_subdirectory(tests)
if (ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_subdirectory(examples)
