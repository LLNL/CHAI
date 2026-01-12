..
    # Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and CHAI
    # project contributors. See the CHAI LICENSE file for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _experimental_design:

*******************
Experimental Design
*******************

CHAI provides data structures that implicitly manage coherence across multiple execution contexts.

-------
Context
-------

Currently, there are two execution contexts that are handled by CHAI. These are represented in the `Context` enum class.
The `HOST` enum value represents synchronous execution on a CPU. The `DEVICE` enum value represents asynchronous execution on a GPU.
Both NVIDIA and AMD GPUs are supported.

--------------
ContextManager
--------------

Implicitly managing data coherence requires managing some global state. This is handled by a singleton called `ContextManager`.
When an application enters an execution context, it uses `ContextManager` to set the current context. `ContextManager` also
tracks which contexts may need synchronization. CHAI data structures can query `ContextManager` to update data coherence and
inform `ContextManager` of needed synchronization or synchronization that has been performed.

Note: It is much faster for `ContextManager` to track synchronization than to repeatedly call `cudaDeviceSynchronize()` or `hipDeviceSynchronize()`.

.. code-block:: cpp

  #include "chai/expt/ContextManager.hpp"
   
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();

  contextManager.setContext(::chai::expt::Context::HOST);
  // Use CHAI data structures in the HOST context...
  contextManager.setContext(::chai::expt::Context::NONE);

  contextManager.setContext(::chai::expt::Context::DEVICE);
  // Use CHAI data structures in the DEVICE context...
  contextManager.setContext(::chai::expt::Context::NONE);
   
------------
ContextGuard
------------

It is easy to forget to reset the current context or even to forget the current context
when writing code. Similar to `std::lock_guard`, CHAI provides `ContextGuard` that sets
the active context and then resets it upon destruction. This is the recommended approach.

.. code-block:: cpp

  #include "chai/expt/ContextGuard.hpp"

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    // Use CHAI data structures in the HOST context...
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::DEVICE};
    // Use CHAI data structures in the DEVICE context...
  }

-----------------
ContextRAJAPlugin
-----------------

In an application that also uses RAJA, CHAI provides a RAJA plugin, `ContextRAJAPlugin`,
that implicitly manages the context in calls to RAJA. To enable this plugin, configure with
`-DCHAI_ENABLE_EXPERIMENTAL_RAJA_PLUGIN=ON` and register the plugin. In the future, registration
may be handled by CHAI.

.. code-block:: cpp

  #include "chai/expt/ContextRAJAPlugin.hpp"
  #include "RAJA/RAJA.hpp"

  static ::RAJA::util::PluginRegistry::add<chai::expt::ContextRAJAPlugin> P(
    "CHAIContextPlugin",
    "Plugin that integrates CHAI context management with RAJA.");
   
  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
    // Use CHAI data structures in the HOST context...
  });

  constexpr int BLOCK_SIZE = 256;

  ::RAJA::forall<::RAJA::cuda_exec_async<BLOCK_SIZE>>(::RAJA::TypedRangeSegment<int>(0, N), [=] __device__ (int i) {
    // Use CHAI data structures in the DEVICE context...
  });
