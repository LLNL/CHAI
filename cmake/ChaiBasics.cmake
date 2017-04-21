set (CUDA_NVCC_FLAGS -std=c++11; --expt-extended-lambda; -G; -g)

#add_definitions(-DDEBUG=0)

set (SPHINX_HTML_THEME "import sphinx_rtd_theme\n
html_theme = \"sphinx_rtd_theme\"\n
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]")
