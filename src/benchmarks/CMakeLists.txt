blt_add_executable(
  NAME arraymanager_benchmarks
  SOURCES chai_arraymanager_benchmarks.cpp
  DEPENDS_ON chai cuda gbenchmark)

blt_add_benchmark(
  NAME arraymanager_benchmarks
  COMMAND arraymanager_benchmarks)

blt_add_executable(
  NAME managedarray_benchmarks
  SOURCES chai_managedarray_benchmarks.cpp
  DEPENDS_ON chai cuda gbenchmark)

blt_add_benchmark(
  NAME managedarray_benchmarks
  COMMAND managedarray_benchmarks)
