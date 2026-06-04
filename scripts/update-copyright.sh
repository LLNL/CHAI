#!/usr/bin/env bash

##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CARE
# contributors. See the CARE LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set -euo pipefail

NEW_END_YEAR="2026"
echo "Updating copyright end year to ${NEW_END_YEAR}:"

for file in "LICENSE" "docs/sphinx/conf.py" "docs/sphinx/conf.py.in" "README.md"; do
    echo "  $file"
    sed -i "s/\([0-9]\{4\}\)-[0-9]\{4\}/\1-${NEW_END_YEAR}/g" "$file"
done
