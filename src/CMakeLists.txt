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
  PUBLIC ${PROJECT_BINARY_DIR}/include)

add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)
