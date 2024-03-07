#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

log(){ printf "\033[32m$@\033[0m\n"; }

wd=$(pwd)
folder=../__amaze_deploy_test__
rm -rvf $folder
mkdir -pv $folder

cd $folder
log Cloning
git clone --depth 1 https://github.com/kgd-al/amaze.git .

python -mvirtualenv .venv
source .venv/bin/activate
log "Created virtual environment"

python -m pip install  --upgrade --no-cache-dir pip setuptools
log "Installed build essentials"

if [ "$1" == "docs" ]
then
    python -m pip install --upgrade --no-cache-dir sphinx readthedocs-sphinx-ext
    log "Installed docs build essentials"
fi

log "Installing package and dependencies"
python -m pip install --upgrade --upgrade-strategy only-if-needed --no-cache-dir $wd[$1]
r=$?
log "Installed package and dependencies: $r"

if [ "$1" == "docs" ]
then
    cd docs/src
    log "Building documentation"
    python -m sphinx -T -W --keep-going -b html -d _build/doctrees -D language=en . html
    log "Documentation built"
    cd -
fi
