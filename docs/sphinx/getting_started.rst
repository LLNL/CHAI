.. getting_started:

===============
Getting Started
===============

This page provides information on how to quickly get up and running with CHAI.

------------
Requirements
------------

------------
Installation
------------

CHAI is hosted in a git repository in the CZ Bitbucket instance. As long as you
are a member of the ``chai`` LC group, you will be able to checkout a copy of
the code.

To clone the repo into your local working space, type:

.. code-block:: bash

  $ git clone --recursive 


The ``--recursive`` argument is required to ensure that the *BLT* submodule is
also checked out. BLT is the build system we use for CHAI, and is available on
GitHub.


^^^^^^^^^^^^^
Building CHAI
^^^^^^^^^^^^^

CHAI uses CMake and BLT to handle builds. Make sure that you have a modern
compiler loaded and the configuration is as simple as:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda ../

CMake will provide output about which compiler is being used, and what version
of CUDA was detected. Once CMake has completed, CHAI can be built with Make:

.. code-block:: bash

  $ make

For more advanced configuration you can use standard CMake variables.

-----------
Basic Usage
-----------

The file ``src/examples/example.cpp`` contains a brief program that shows how
CHAI can be used. Let's walk through this example, line-by-line:

.. literalinclude:: ../../src/examples/example.cpp
  :language: c++


