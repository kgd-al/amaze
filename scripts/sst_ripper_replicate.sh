#!/bin/bash

datadir=$1
[ -z "$datadir" ] && datadir=remote/dd

outdir=$2
[ -z "$outdir" ] && outdir=tmp/reevals

reevals=$(grep '^> M' $datadir/*/*/*/log | wc -l)
echo "Reevals: $reevals"

for model in $datadir/*/*/*/best_model.zip
do
    for maze in $(grep '^> M' $(dirname $model)/log | cut -d ' ' -f2-)
    do
        odir=$outdir/$(dirname $model)/$maze
        if [ ! -d $odir ]
        then
            mkdir -p $odir
            ./bin/maze_viewer.py --maze $maze --controller $model --eval $odir --trajectory > $odir/log 2>&1
        fi
        echo "$maze $model"
    done
done | awk -vs=$reevals '{printf "[%5.1f%%] %s\t\t\r", 100*NR/s, $1}END{print "\nDone"}'
# done | pv -ls $reevals > /dev/null
