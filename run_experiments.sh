#!/bin/bash
seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
run=1
echo $seeds
for s in ${seeds[*]}; do
    CUDA_VISIBLE_DEVICES=1 python3 -m a2 > out.log
    run=$((run + 1))
done
