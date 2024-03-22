#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

_log="no-log"
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

cols=40 #$(($(tput cols) - 2))
short_output(){
    tee -a $_log - 2>&1 | stdbuf -o0 sed "s|^\(.\{$cols\}\).*|\1\r|" | stdbuf -o0 tr -d "\n"
    echo
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

    line

    folder=../__amaze_deploy_test__/$type_name

    rm -rf $folder
    mkdir -pv $folder
    folder=$(realpath $wd/$folder)

    _log=$folder.log

    log 35 "deploy folder = $folder"
    log 35 "Processing spec '$spec'"

    install(){
        python -m pip download -d $download_cache $@ | short_output
        python -m pip install --upgrade --find-links=$download_cache $@ | short_output
    }

    git ls-files | tar Tc - | tar Cx $folder
    cd $folder
#     git clone --depth 1 https://github.com/kgd-al/amaze.git .

    date > $_log
    log Copied

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

    log "Installing package and dependencies for spec '$spec'"
    install .$spec
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
        pytest | short_output
    fi

    deactivate
    cd -
    line
}

for type in '' 'full' 'tests' 'docs'
# for type in 'docs'
do
    deploy "$type"
done
