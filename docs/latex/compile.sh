#!/bin/bash

base=$(dirname $0)
base=$(realpath $base)
cd $base
log=$base/.log
date > $log

compile(){
    cd $1

    mode=$2
    echo "#### mode: $mode" | tee -a $log
    for i in 0 1
    do
        pdflatex --shell-escape --interaction=nonstopmode --jobname=$mode *.tex >> $log
    done
    convert -density 300 $mode.pdf $mode.png

    rm *.{aux,log}

    cd ..
}

compile_ld(){
    echo "### Compiling $1 figure" | tee -a $log
    for mode in light dark
    do
        compile $1 $mode
    done
}

compile_ld maze
compile_ld agents
compile_ld complexity

compile maze light-wide

montage -geometry +100+0 agents/light-{1,3}.png agents/light-1-3.png
