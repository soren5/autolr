#!/bin/bash
seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
run=1
echo $seeds
for s in ${seeds[*]}; do
    python -m main --seed $s --parameters=parameters/fake_fitness.yml --run $run
    run=$((run + 1))
done
