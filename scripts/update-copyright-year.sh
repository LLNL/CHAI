#!/usr/bin/env zsh
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
# This is used for the ~*tpl* line to ignore files in bundled tpls
setopt extended_glob

autoload colors

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

files_no_license=$(grep -l '2016-19,' \
  benchmarks/**/*(^/) \
  cmake/**/*(^/) \
  docs/**/*~*rst(^/)\
  examples/**/*(^/) \
  scripts/**/*~*copyright*(^/) \
  src/**/*~*tpl*(^/) \
  tests/**/*(^/) \
  CMakeLists.txt)

if [ $files_no_license ]; then
  print "${RED} [!] Some files need copyright year updating: ${NOCOLOR}"
  echo "${files_no_license}"

  echo ${files_no_license} | xargs sed -i '' 's/2016-19,/2016-20,/'

  print "${GREEN} [Ok] Copyright years updated."

  exit 0
else
  print "${GREEN} [Ok] All files have required license info."
  exit 0
fi
