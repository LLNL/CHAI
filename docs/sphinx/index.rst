******
CHAI
******

CHAI is a C++ libary providing an array object that can be used in multiple
memory spaces. Data is automatically migrated based on copy-construction,
allowing for transparent data access regardless of location. CHAI can be used
standalone, but is best when paired with the RAJA library, which has build in
CHAI integration that takes care of everything.


.. toctree::
	:maxdepth: 2
	:caption: Basics

	getting_started
	tutorial
	known_issues

.. toctree::
	:maxdepth: 2
	:caption: Reference

	configuration
  concepts
  code_documentation

.. toctree::
	:maxdepth: 2
	:caption: Contributing

	contribution_guide
	developer_guide
