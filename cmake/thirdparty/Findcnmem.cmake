find_path(cnmem_INCLUDE_DIRS
  cnmem.h
  HINTS ${cnmem_DIR}/include)

find_library(cnmem_LIBRARIES
  NAMES cnmem
  HINTS ${cnmem_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cnmem
  "Failed to find cnmem"
  cnmem_INCLUDE_DIRS
  cnmem_LIBRARIES)
