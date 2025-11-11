#!/bin/bash

cd joss/inara
make pdf ARTICLE=../paper.md TARGET_FOLDER=.. || exit 1

pgrep -f paper.pdf >/dev/null || xdg-open ../paper.pdf
