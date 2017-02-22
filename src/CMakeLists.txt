set (chai_headers
  ExecutionSpaces.hpp
  ManagedArray.hpp
  ResourceManager.hpp)

set (chai_sources
  ManagedArray.cpp
  ResourceManager.cpp)

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  USE_CUDA True)
