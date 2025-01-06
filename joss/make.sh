#!/bin/bash

cd joss/inara
make pdf ARTICLE=../paper.md || exit 1

pgrep -f paper.pdf >/dev/null || xdg-open publishing-artifacts/paper.pdf
