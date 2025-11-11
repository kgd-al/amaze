#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit

eval "$(pyenv init -)"

folder=$(realpath $(dirname $0))
tmp=$folder/___cache___/
mkdir -p $tmp
downloads=$tmp/downloads
mkdir -p $downloads
log=$tmp/log

line(){
    n=$(tput cols)
    printf "=%.0s" $(seq $n)
    printf "\n"
}
export -f line

amaze_root=.
while [ ! -f $amaze_root/pyproject.toml ]
do
    amaze_root=$amaze_root/..
done
amaze_root=$(realpath $amaze_root)
echo "Using local amaze version at $amaze_root"

workers=(
    "urlb 3.8 https://github.com/rll-research/url_benchmark;numpy==1.19.2"
    "gymretro 3.8 gym-retro;gym==0.25.2"
    "amaze 3.10 $amaze_root" #amaze-benchmarker"
    "gymnasium 3.10 numpy<2;mujoco<3;gymnasium[classic-control,box2d,mujoco,atari,accept-rom-license]"
    "procgen 3.10 procgen"
    "metaworld 3.10 git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld"
    "levdoom 3.10 https://github.com/TTomilin/LevDoom;opencv-python>=3.0;gymnasium==0.28.1"
    "lab2d 3.10 dmlab2d"
    "labmaze 3.10 labmaze"
    "mazeexplorer 3.10 git+https://github.com/microsoft/MazeExplorer"
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
    read worker python_version package <<< $data
    venv=$tmp/venvs/__$worker
    install_ok=$venv/install_ok

    git_repo=""
    [ ${package:0:5} == "https" ] && git_repo="git_repo"

    pyenv shell $python_version
    python=python$python_version

    echo "worker: $worker ($(python --version) - $($python --version))"

    if [ ! -d $venv ] || [ ! -f $install_ok ]
    then
        [ -d $venv ] && rm -rf $venv

        $python -m venv $venv
        source $venv/bin/activate

        if [ -z $git_repo ]
        then
          packages=$(tr ';' ' ' <<< $package)
          echo "packages: $packages"

#          set -x
          $python -m pip download -d $downloads $packages --exists-action i #| short_output
          $python -m pip install --upgrade --find-links=$downloads $packages $pretty_packages \
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

          if [ "$worker" == "urlb" ]
          then
            echo "Creating package structure"

            echo "Generating setup.py"
            yaml_packages=$(tail -n+17 conda_env.yml | grep -v 'pip:' | sed -e "s/.* - //" -e 's/.*dm_control.git.*/dm_control/' -e 's/mujoco_py.*/mujoco_py==2.1.2.14/')
            packages="$packages $yaml_packages"
            cat <<- EOF > setup.py
from setuptools import setup, find_packages

setup(
    name='urlb',
    url='$git_repo',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[],
)
EOF
          fi

          pip install . --upgrade --find-links=$downloads $packages $pretty_packages \
#           | short_output
          pip uninstall $worker -y
        fi

        if [ "$worker" == "gymretro" ]
        then
            python3 -m retro.import $(dirname $0)/gym-retro-roms | short_output
        fi

        pip list
        touch $install_ok

    else
        source $venv/bin/activate
    fi

    OLD_PYTHONPATH=${PYTHONPATH+""}
    [ -n "$git_repo" ] && export PYTHONPATH="$OLD_PYTHONPATH:$venv/$git_repo"

    $python $folder/generate_one.py $worker

    export PYTHONPATH="$OLD_PYTHONPATH"

done
line

unset PYENV_VERSION
