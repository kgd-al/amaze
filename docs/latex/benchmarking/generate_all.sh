#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

folder=$(dirname $0)
tmp=$folder/___cache___/
downloads=$tmp/downloads
log=$tmp/log

workers=(
    "amaze amaze-benchmarker"
    "gymnasium gymnasium[classic-control,box2d,mujoco,atari,accept-rom-license]"
)

if [ $# -ge 1 ]
then
    echo "Detailed!"
    workers+=(
        "procgen procgen"
    )
fi

cols=$(($(tput cols) - 2))
short_output(){
    tee -a $log - 2>&1 | stdbuf -o0 awk -vc=$cols '
        {
            printf "%s", substr($0, 0, c);
            for (i=length; i<c; i++) printf " ";
            printf "\r";
        }'
    printf " %.0s" $(seq $cols)
    printf "\r"
}

date > $log

for data in "${workers[@]}"
do
    line | tee -a $log
    read worker package <<< $data
    echo "worker: $worker"
    venv=$tmp/venvs/__$worker

    if [ ! -d $venv ]
    then
        python -m venv $venv
        source $venv/bin/activate

        packages=$(tr ';' ' ' <<< $package)
        echo "package: $package"

        python -m pip download -d $downloads $packages | short_output
        python -m pip install --upgrade --find-links=$downloads $packages pandas rich \
        | short_output

        pip list

    else
        source $venv/bin/activate
    fi

    $folder/generate_one.py $worker $@

done
line
