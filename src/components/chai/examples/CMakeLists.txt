blt_add_executable(
  NAME chai-example.exe
  SOURCES example.cpp
  DEPENDS_ON chai
  USE_CUDA True)
