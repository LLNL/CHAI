set (chai_headers
  ChaiMacros.hpp
  ExecutionSpaces.hpp
  ManagedArray.hpp
  ManagedArray.inl
  ResourceManager.hpp
  ResourceManager.inl
  )

set (chai_sources
  ResourceManager.cpp)

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  USE_CUDA True)

add_subdirectory(tests)
add_subdirectory(examples)
