blt_add_executable(
  NAME allocate_benchmark
  SOURCES chai_allocate_benchmark.cpp
  DEPENDS_ON chai cuda gbenchmark)

blt_add_benchmark(
  NAME allocate_benchmark
  COMMAND allocate_benchmark)
