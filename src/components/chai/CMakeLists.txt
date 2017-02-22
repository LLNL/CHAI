set (chai_headers
  ChaiMacros.hpp
  ExecutionSpaces.hpp
  ManagedArray.cpp
  ManagedArray.hpp
  ResourceManager.hpp)

set (chai_sources
  ManagedArray.cpp
  ResourceManager.cpp)

set_source_files_properties(
  ManagedArray.cpp
  PROPERTIES HEADER_FILE_ONLY True)

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  HEADERS_OUTPUT_SUBDIR chai
  USE_CUDA True)

add_subdirectory(tests)
add_subdirectory(examples)
