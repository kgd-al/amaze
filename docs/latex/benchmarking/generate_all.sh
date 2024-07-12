#!/bin/bash

folder=$(dirname $0)
tmp=$folder/___cache___/
downloads=$tmp/downloads
log=$tmp/log

workers=(
    "amaze amaze-benchmarker"
    "gymnasium gymnasium"
)

cols=$(($(tput cols) - 2))
short_output(){
    tee -a $log - 2>&1 | stdbuf -o0 awk -vc=$cols '{printf "%s\r", substr($0, 0, c);}'
}

date > $log

for data in "${workers[@]}"
do
    line | tee -a $log
    read worker package <<< $data
    echo "worker: $worker"
    echo "package: $package"
    venv=$tmp/venvs/__$worker
    python -m venv $venv
    source $venv/bin/activate

    python -m pip download -d $downloads $package | short_output
    python -m pip install --upgrade --find-links=$downloads $package | short_output

    pip list

done
line
