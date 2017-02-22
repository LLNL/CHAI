blt_add_executable(
  NAME chai-example.exe
  SOURCES example.cpp
  DEPENDS_ON chai
  USE_CUDA True)

target_include_directories(
  chai-example.exe
  PUBLIC ${PROJECT_SOURCE_DIR}/src)
