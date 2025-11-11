#!/bin/bash

# == Failure handling (fast-fail and warnings)

set -euo pipefail
shopt -s inherit_errexit

ok=1
check(){
  if [ $ok -ne 0 ]
  then
    cat $_log
    printf "\033[31mPackage is not ready to deploy."
    printf " See error(s) above.\033[0m\n"
  else
    printf "\033[32mPackage checks out.\033[0m\n"
  fi
}
trap check exit

# == Prepare work place

wd=$(pwd)
base=../__amaze_deploy_test__/
mkdir -pv $base
download_cache=$(realpath $base/download_cache)
mkdir -pv $download_cache

_log="$base/global.log"
log(){
    if [ $# -gt 1 ]
    then
        color=$1
        shift
    else
        color=32
    fi
    printf "\033[${color}m$1\033[0m\n" | tee -a $_log
}

log "$(date)"
log 35 "working directory: $wd"
log 35 "download cache = $download_cache"

cols=$(($(tput cols) - 2))
short_output(){
    tee -a $_log - 2>&1 | stdbuf -o0 awk -vc=$cols '
        {
            printf "%s", substr($0, 0, c);
            for (i=length; i<c; i++) printf " ";
            printf "\r";
        }'
    printf " %.0s" $(seq $cols)
    printf "\r"
}

# == Sanity checks

if grep -rn 'from amaze' src
then
  log 31 "Absolute imports in sources"
  exit 2
fi

black src tests examples
flake8 src tests examples

# Worker
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

    folder=$base/$type_name

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

    python -m venv _venv
    source _venv/bin/activate
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

    pip list | tee -a $_log

    if [ "$type" == "docs" ]
    then
        log "Building documentation"
        ./commands.sh docs 2>&1 | short_output
        log "Documentation built"
        cd -
    elif [[ "$type" =~ "tests" ]]
    then
        pytest --small-scale --test-examples --test-extension sb3
    fi

    deactivate
    cd $wd

    rm -rf $folder && echo "Cleared test folder" | tee -a $_log

    line
}

# Work
for type in '' 'full' 'docs' 'tests,dev'
#for type in 'tests,dev'
do
    deploy "$type"
done

# Got to the end without error!
ok=0

