******
CHAI
******

CHAI is a C++ libary providing an array object that can be used transparently
in multiple memory spaces. Data is automatically migrated based on
copy-construction, allowing for transparent data access regardless of location.
CHAI can be used standalone, but is best when paired with the RAJA library,
which has build in CHAI integration that takes care of everything.

- If you want to get and install CHAI, take a look at our getting started guide. 
- If you are looking for documentation about a particular CHAI function, see the code documentation.
- Want to contribute? Take a look at our developer and contribution guides.

Any questions? Contact chai-dev@llnl.gov

.. toctree::
  :maxdepth: 2
  :caption: Basics

  getting_started
  tutorial
  known_issues

.. toctree::
  :maxdepth: 2
  :caption: Reference

  concepts
  code_documentation

.. toctree::
  :maxdepth: 2
  :caption: Contributing

  contribution_guide
  developer_guide
