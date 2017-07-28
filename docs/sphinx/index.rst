.. Copyright (c) 2016, Lawrence Livermore National Security, LLC. All
 rights reserved.
 
 Produced at the Lawrence Livermore National Laboratory
 
 This file is part of CHAI.
 
 LLNL-CODE-705877
 
 For details, see https:://github.com/LLNL/CHAI
 Please also see the NOTICE and LICENSE files.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 
 - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.
 
 - Neither the name of the LLNS/LLNL nor the names of its contributors
   may be used to endorse or promote products derived from this
   software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

******
CHAI
******

CHAI is a libary handling automatic data migration to different memory spaces
behind an array-style interface.

CHAI Quickstart Guide
=====

This guide provides information on how to quickly get up and running with CHAI!

Downloading CHAI
-----

CHAI is hosted in a git repository in the CZ Bitbucket instance. As long as are
a member of the ``chai`` LC group, you will be able to checkout a copy of the code.

To clone the repo into your local working space, type:

.. code-block:: bash

  $ git clone --recursive 


The ``--recursive`` argument is required to ensure that the *BLT* submodule is
also checked out. BLT is the build system we use for CHAI, and is available on
GitHub.


Building CHAI
-----

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

Example
-------

The file ``src/examples/example.cpp`` contains a brief program that shows how
CHAI can be used. Let's walk through this example, line-by-line:

.. literalinclude:: ../../src/examples/example.cpp
  :language: c++
