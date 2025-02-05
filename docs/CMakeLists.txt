##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif ()

if (SPHINX_FOUND)
  set (SPHINX_HTML_THEME "import sphinx_rtd_theme\n
html_theme = \"sphinx_rtd_theme\"\n
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]")

  add_subdirectory(sphinx)
endif ()
