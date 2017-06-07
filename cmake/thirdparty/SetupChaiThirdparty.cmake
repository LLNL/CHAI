message (STATUS "cnmem support is ${ENABLE_CNMEM}")

if (ENABLE_CNMEM)
  if (NOT ENABLE_CUDA)
    message(FATAL_ERROR "Cannot use cnmem without CUDA. Please re-configure with -DENABLE_CUDA=On")
  endif ()

  include(${CMAKE_SOURCE_DIR}/cmake/thirdparty/Findcnmem.cmake)

  blt_register_library(
    NAME cnmem
    INCLUDES ${cnmem_INCLUDE_DIRS}
    LIBRARIES ${cnmem_LIBRARIES})
endif()
