#!/usr/bin/env zsh

##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

# This is used for the ~*tpl* line to ignore files in bundled tpls
setopt extended_glob

autoload colors

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

LIC_CMD=$(which lic)
if [ ! $LIC_CMD ]; then
  echo "${RED} [!] This script requires the lic command.${NOCOLOR}"
  exit 255
fi

echo "Applying licenses to files"

files_no_license=$(grep -rL "SPDX-License-Identifier: BSD-3-Clause" . \
   --exclude-dir=.git \
   --exclude-dir=blt \
   --exclude-dir=umpire \
   --exclude-dir=raja \
   --exclude-dir=radiuss-spack-configs \
   --exclude-dir=uberenv \
   --exclude=.gitmodules \
   --exclude=.mailmap \
   --exclude=LICENSE \
   --exclude=COPYRIGHT \
   --exclude=NOTICE \
   --exclude=*.json)

echo $files_no_license | xargs $LIC_CMD -f scripts/license.txt 

echo "${GREEN} [Ok] License text applied.${NOCOLOR}"
