..
    # Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
    # project contributors. See the CHAI LICENSE file for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _getting_started:

===============
Getting Started
===============

This page provides information on how to quickly get up and running with CHAI.

------------
Installation
------------

CHAI is hosted on GitHub `here <https://github.com/LLNL/CHAI>`_.  To clone the
repo into your local working space, type:

.. code-block:: bash

  $ git clone --recursive git@github.com:LLNL/CHAI.git


The ``--recursive`` argument is required to ensure that the *BLT* submodule is
also checked out. `BLT <https://github.com/LLNL/BLT>`_ is the build system we
use for CHAI.


^^^^^^^^^^^^^
Building CHAI
^^^^^^^^^^^^^

CHAI uses CMake and BLT to handle builds. Make sure that you have a modern
compiler loaded and the configuration is as simple as:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ../

By default, CHAI will attempt to build with CUDA. CMake will provide output
about which compiler is being used, and what version of CUDA was detected. Once
CMake has completed, CHAI can be built with Make:

.. code-block:: bash

  $ make

For more advanced configuration, see :doc:`advanced_configuration`.

-----------
Basic Usage
-----------

Let's take a quick tour through CHAI's most important features. A complete
listing you can compile is included at the bottom of the page. First, let's
create a new ManagedArray object. This is the interface through which you will
want to access data:

.. code-block:: cpp

  chai::ManagedArray<double> a(100);

This creates a ManagedArray storing elements of type double, with 100 elements
allocated in the CPU memory.

Next, let's assign some data to this array. We'll use CHAI's forall helper
function for this, since it interacts with the ArrayManager for us to ensure
the data is in the appropriate ExecutionSpace:

.. code-block:: cpp

  forall(sequential(), 0, 100, [=] (int i) {
    a[i] = 3.14 * i;
  });

CHAI's ArrayManager can copy this array to another ExecutionSpace
transparently. Let's use the GPU to double the contents of this array:

.. code-block:: cpp

  forall(cuda(), 0, 100, [=] __device__ (int i) {
    a[i] = 2.0 * a[i];
  });

We can access the array again on the CPU, and the ArrayManager will handle
copying the modified data back:

.. code-block:: cpp
  
  forall(sequential(), 0, 100, [=] (int i) {
    std::cout << "a[" << i << "] = " << a[i] << std::endl;
  });

