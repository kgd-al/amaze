#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

folder=$(realpath $(dirname $0))
tmp=$folder/___cache___/
downloads=$tmp/downloads
log=$tmp/log

workers=(
    "amaze amaze-benchmarker"
    "gymnasium numpy<2;mujoco<3;gymnasium[classic-control,box2d,mujoco,atari,accept-rom-license]"
    "procgen procgen"
    "metaworld git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld"
    "levdoom https://github.com/TTomilin/LevDoom;opencv-python>=3.0"
    "lab2d dmlab2d"
    "labmaze labmaze"
    "mazeexplorer git+https://github.com/microsoft/MazeExplorer"
)
pretty_packages="pandas rich"

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
    install_ok=$venv/install_ok

    git_repo=""
    [ ${package:0:5} == "https" ] && git_repo="git_repo"

    if [ ! -d $venv ] || [ ! -f $install_ok ]
    then
        [ -d $venv ] && rm -rf $venv

        python -m venv $venv
        source $venv/bin/activate

        if [ -z $git_repo ]
        then
          packages=$(tr ';' ' ' <<< $package)
          echo "packages: $packages"

#          set -x
          python -m pip download -d $downloads $packages --exists-action i #| short_output
          python -m pip install --upgrade --find-links=$downloads $packages $pretty_packages \
          #| short_output

        else
          cd $venv
          packages=$(tr ';' ' ' <<< $package)
          git=$(cut -d ' ' -f 1 <<< $packages)
          packages=$(cut -d ' ' -f 2- <<< $packages)
          echo "git: $git"
          echo "packages: $packages"
          git clone $git $git_repo
          cd $git_repo
          pip install . --upgrade --find-links=$downloads $packages $pretty_packages \
          | short_output
          pip uninstall $worker -y
        fi

        pip list
        touch $install_ok

    else
        source $venv/bin/activate
    fi

    OLD_PYTHONPATH=${PYTHONPATH+""}
    [ -n "$git_repo" ] && export PYTHONPATH="$OLD_PYTHONPATH:$venv/$git_repo"

    $folder/generate_one.py $worker

    export PYTHONPATH="$OLD_PYTHONPATH"

done
line
