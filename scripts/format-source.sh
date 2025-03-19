#!/usr/bin/env bash
##############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and CHAI
# project contributors. See the CHAI LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

find . -type f -iname '*.hpp' -o -iname '*.cpp' | grep -v -e blt -e tpl | xargs clang-format -i
