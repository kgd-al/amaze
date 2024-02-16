#!/bin/bash

base=$(dirname $0)
base=$(realpath $base)
cd $base
log=$base/.log
date > $log

compile(){
    cd $1
    echo "### Compiling $1 figure" | tee -a $log
    for mode in light dark
    do
        echo "#### mode: $mode" | tee -a $log
        for i in 0 1
        do
            pdflatex --shell-escape --interaction=nonstopmode --jobname=$mode *.tex >> $log
        done
        convert -density 300 $mode.pdf $mode.png
    done

    rm *.{aux,log,pdf}

    cd ..
}

compile maze
compile agents
