..
    # Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
    # project contributors. See the CHAI LICENSE file for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _advanced_configuration:

================
Configuring CHAI
================

In addition to the normal options provided by CMake, CHAI uses some additional
configuration arguments to control optional features and behavior. Each
argument is a boolean option, and  can be turned on or off:

    -DENABLE_CUDA=Off

Here is a summary of the configuration options, their default value, and meaning:

      ===========================  ======== ===============================================================================
      Variable                     Default  Meaning
      ===========================  ======== ===============================================================================
      ENABLE_CUDA                  Off      Enable CUDA support.
      ENABLE_HIP                   Off      Enable HIP support.
      CHAI_ENABLE_GPU_SIMULATION_MODE   Off      Simulates GPU execution.
      CHAI_ENABLE_UM                    Off      Enable support for CUDA Unified Memory.
      CHAI_ENABLE_IMPLICIT_CONVERSIONS  On       Enable implicit conversions between ManagedArray and raw pointers
      CHAI_ENABLE_MANAGER          On       Enable the ArrayManager.
      ENABLE_TESTS                 On       Build test executables.
      ENABLE_BENCHMARKS            On       Build benchmark programs.
      ===========================  ======== ===============================================================================

These arguments are explained in more detail below:

* ENABLE_CUDA
  This option enables support for GPUs using CUDA. If CHAI is built without CUDA, HIP, or
  GPU_SIMULATION_MODE support, then only the ``CPU`` execution space is available for use.

* ENABLE_HIP
  This option enables support for GPUs using HIP. If CHAI is built without CUDA, HIP, or
  GPU_SIMULATION_MODE support, then only the ``CPU`` execution space is available for use.

* CHAI_ENABLE_GPU_SIMULATION_MODE
  This option simulates GPU support by enabling the GPU execution space, backed by a HOST
  umpire allocator. If CHAI is built without CUDA, HIP, or GPU_SIMULATION_MODE support, 
  then only the ``CPU`` execution space is available for use.

* CHAI_ENABLE_UM
  This option enables support for Unified Memory as an optional execution
  space. When a ``ManagedArray`` is allocated in the ``UM`` space, CHAI will
  not manually copy data. Data movement in this case is handled by the CUDA
  driver and runtime.

* CHAI_ENABLE_IMPLICIT_CONVERSIONS
  This option will allow implicit casting between an object of type
  ``ManagedArray<T>`` and the correpsonding raw pointer type ``T*``. This
  option is disabled by default, and should be used with caution.

* CHAI_ENABLE_MANAGER
  This option enables usage of the ``ArrayManager`` class. Turning it off lets
  the ``ManagedArray`` objects function as thin wrappers around a raw pointer.
  The thin wrapper version can be used with CPU-only allocations, unified
  memory, or with architectures using a single memory space.

* ENABLE_TESTS
  This option controls whether or not test executables will be built.

* ENABLE_BENCHMARKS
  This option will build the benchmark programs used to test ``ManagedArray``
  performance.

