#!/bin/bash

set -euo pipefail
shopt -s inherit_errexit
export LC_ALL=C

outfolder=$1
[ -z "$outfolder" ] && outfolder=stash/dd

datafile=$(dirname $0)/mazes.dat
if [ ! -f "$datafile" ]
then
    echo "Could not find class specifications file at '$datafile'" >&2
    exit 1
fi

while read name maze
do
    [[ $name =~ "#" ]] && continue
    echo "$name: $maze"

    for j in 0 #{0..9}
    do
        for i in {0..9}
        do
            for trainer in A2C # DQN PPO
            do
                basedir=$outfolder/$trainer/$name
                id=run$j$i
                odir=$basedir/$id
                [ -d "$odir" ] && continue

                train_maze=$(sed "s/M16/M$j/" <<< $maze)
                test_maze=$(sed "s/M16/M1$j/" <<< $maze)
                nice ./bin/sb3/trainer.py --id $id --seed $i --trainer $trainer \
                    --folder $basedir --overwrite ABORT --budget 1000000 --evals 400 \
                    --maze $train_maze --eval-maze $test_maze --all-permutations
            done
        done
    done
done < $datafile
