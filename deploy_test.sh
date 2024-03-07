#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

_log=
log(){
    if [ $# -gt 1 ]
    then
        color=$1
        shift
    else
        color=32
    fi
    printf "\033[${color}m$1\033[0m\n" | tee $_log
}

cols=$(tput cols)
short_output(){
    tee -a $_log - | cut -c -$cols | tr "\n" "\r"
}

wd=$(pwd)
download_cache=$(realpath $wd/../__amaze_deploy_test_download_cache__)
mkdir -pv $download_cache
log "$(date)"
log 35 "working directory: $wd"
log 35 "download cache = download_cache"

deploy(){
    type=$1
    type_name=$(tr "," "_" <<< $type)
    if [ -z "$type_name" ]
    then
        type_name=default
        spec=""
    else
        spec="[$type]"
    fi

    folder=$(realpath $wd/../__amaze_deploy_test__/$type_name)
    log 35 "deploy folder = $folder"
    rm -rf $folder
    mkdir -pv $folder

    _log=$folder/log

    install(){
        python -m pip download -d $download_cache $@ | short_output
        python -m pip install --upgrade --find-links=$download_cache $@ | short_output
    }

    cd $folder
    git clone --depth 1 https://github.com/kgd-al/amaze.git .

    date > $_log
    log Cloned

    python -mvirtualenv .venv
    source .venv/bin/activate
    log "Created virtual environment"

    install pip setuptools
    log "Installed build essentials"

    if [ "$type" == "docs" ]
    then
        install sphinx readthedocs-sphinx-ext
        log "Installed docs build essentials"
    fi

    log "Installing package and dependencies"
    install $wd$spec
    r=$?
    log "Installed package and dependencies: $r"

    if [ "$type" == "docs" ]
    then
        cd docs/src
        log "Building documentation"
        python -m sphinx -T -W --keep-going -b html -d _build/doctrees -D language=en . html | short_output
        log "Documentation built"
        cd -
    elif [ "$type" == "tests" ]
    then
        pytest
    fi

    deactivate
}

for type in '' 'full' 'tests' 'docs'
do
    deploy "$type"
done
