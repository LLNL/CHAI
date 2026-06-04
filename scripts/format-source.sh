#!/usr/bin/env bash
##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

find . -type f -iname '*.hpp' -o -iname '*.cpp' | grep -v -e blt -e tpl | xargs clang-format -i
