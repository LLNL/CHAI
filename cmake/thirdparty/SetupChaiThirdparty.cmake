message (STATUS "cnmem support is ${ENABLE_CNMEM}")

if (ENABLE_CNMEM)
  include(${CMAKE_SOURCE_DIR}/cmake/thirdparty/Findcnmem.cmake)

  blt_register_library(
    NAME cnmem
    INCLUDES ${cnmem_INCLUDE_DIRS}
    LIBRARIES ${cnmem_LIBRARIES})
endif()
