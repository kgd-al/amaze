#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

wd=$(pwd)
folder=../__amaze_deploy_test__
rm -rvf $folder
mkdir -pv $folder

cd $folder
git clone --depth 1 https://github.com/kgd-al/amaze.git .

python -mvirtualenv .venv
python -m pip install  --upgrade --no-cache-dir pip setuptools

if [ "$1" == "docs" ]
then
    python -m pip install --upgrade --no-cache-dir sphinx readthedocs-sphinx-ext
fi

python -m pip install --upgrade --upgrade-strategy only-if-needed --no-cache-dir $wd[$1]

if [ "$1" == "docs" ]
then
    cd docs/src
    python -m sphinx -T -W --keep-going -b html -d _build/doctrees -D language=en . html
    cd -
fi
