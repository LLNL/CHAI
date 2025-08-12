..
    # Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
    # project contributors. See the CHAI LICENSE file for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _developer_guide:

===============
Developer Guide
===============

CHAI shares its Uberenv workflow with other projects. The documentation is
therefore `shared`_.

.. shared: <https://radiuss-ci.readthedocs.io/en/latest/uberenv.html#uberenv-guide)

This page will provides some CHAI specific examples to illustrate the
workflow described in the documentation.

Machine specific configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  $ ls -c1 scripts/uberenv/spack_configs
  blueos_3_ppc64le_ib
  darwin
  toss_3_x86_64_ib
  blueos_3_ppc64le_ib_p9
  config.yaml

CHAI has been configured for ``toss_3_x86_64_ib`` and other systems.

Vetted specs
^^^^^^^^^^^^

.. code-block:: bash

  $ ls -c1 .gitlab/*jobs.yml
  .gitlab/lassen-jobs.yml
  .gitlab/dane-jobs.yml

CI contains jobs for dane.

.. code-block:: bash

  $ git grep -h "SPEC" .gitlab/dane-jobs.yml | grep "gcc"
      SPEC: "%gcc@4.9.3"
      SPEC: "%gcc@6.1.0"
      SPEC: "%gcc@7.1.0"
      SPEC: "%gcc@7.3.0"
      SPEC: "%gcc@8.1.0"

We now have a list of the specs vetted on ``dane``/``toss_3_x86_64_ib``.

.. note::
  In practice, one should check if the job is not *allowed to fail*, or even deactivated.

MacOS case
^^^^^^^^^^

In CHAI, the Spack configuration for MacOS contains the default compilers depending on the OS version (`compilers.yaml`), and a commented section to illustrate how to add `CMake` as an external package. You may install CMake with homebrew, for example.


Using Uberenv to generate the host-config file
----------------------------------------------

We have seen that we can safely use `gcc@8.1.0` on dane. Let us ask for the default configuration first, and then ask for RAJA support and link to develop version of RAJA:

.. code-block:: bash

  $ python scripts/uberenv/uberenv.py --spec="%clang@9.0.0"
  $ python scripts/uberenv/uberenv.py --spec="%clang@9.0.0+raja ^raja@develop"

Each will generate a CMake cache file, e.g.:

.. code-block:: bash

  hc-dane-toss_3_x86_64_ib-clang@9.0.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake

Using host-config files to build CHAI
-------------------------------------

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -C <path_to>/<host-config>.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

It is also possible to use this configuration with the CI script outside of CI:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake scripts/gitlab/build_and_test.sh

Testing new dependencies versions
---------------------------------

CHAI depends on Umpire, and optionally CHAI. Testing with newer versions of both is made straightforward with Uberenv and Spack:

* ``$ python scripts/uberenv/uberenv.py --spec=%clang@9.0.0 ^umpire@develop``
* ``$ python scripts/uberenv/uberenv.py --spec=%clang@9.0.0+raja ^raja@develop``

Those commands will install respectively `umpire@develop` and `raja@develop` locally, and generate host-config files with the corresponding paths.

Again, the CI script can be used directly to install, build and test in one command:

.. code-block:: bash

  $ SPEC="%clang@9.0.0 ^umpire@develop" scripts/gitlab/build_and_test.sh
