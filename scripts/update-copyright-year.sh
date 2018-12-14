#!/usr/bin/env zsh
#######################################################################
# Copyright (c) 2016-2018, Lawrence Livermore National Security, LLC. All
# rights reserved.
# 
# Produced at the Lawrence Livermore National Laboratory.
# 
# This file is part of CHAI.
# 
# LLNL-CODE-705877
# 
# For details, see https:://github.com/LLNL/CHAI
# Please also see the NOTICE and LICENSE files.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the
#   distribution.
# 
# - Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#######################################################################

# This is used for the ~*tpl* line to ignore files in bundled tpls
setopt extended_glob

autoload colors

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

files_no_license=$(grep -l '2016,' \
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

  echo ${files_no_license} | xargs sed -i '' 's/2016,/2016-2018,/'

  print "${GREEN} [Ok] Copyright years updated."

  exit 0
else
  print "${GREEN} [Ok] All files have required license info."
  exit 0
fi
