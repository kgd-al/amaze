#!/bin/bash

workers=(
    "amaze amaze-benchmarker"
    "gymnasium gymnasium"
)

for data in "${workers[@]}"
do
    line
    read worker package <<< $data
    echo "worker: $worker"
    echo "package: $package"
done
line
