#!/usr/bin/env bash

find . -type f -iname '*.hpp' -o -iname '*.cpp' | grep -v -e blt -e tpl | xargs clang-format -i
