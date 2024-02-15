#!/bin/bash

base=$(dirname $0)
base=$(realpath $base)
cd $base
log=$base/.log
date > $log

compile(){
    pdflatex --shell-escape --interaction=errorstopmode $@
}

cd maze
echo "### Compiling maze figure" | tee -a $log
echo "#### Dark mode 1" | tee -a $log
compile --jobname=maze_dark maze.tex <<< "x\n" #>> $log
echo "#### Dark mode 2" | tee -a $log
compile --jobname=dark maze.tex <<< "x\n" #>> $log
echo "#### Dark mode 3" | tee -a $log
pdflatex --shell-escape --interaction=errorstopmode --jobname=dark maze.tex <<< "x\n" #>> $log
# echo "#### Light mode" | tee -a $log
# compile --jobname=maze_light maze.tex >> $log
