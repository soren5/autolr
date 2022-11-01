#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --job-name=sending_cif
#SBATCH --output=sending_cif%j.log
for filename in ./parameters/many_runs/cif/*; 
do(
    [ -e "$filename" ] || continue 
    echo "sending: $filename"
    sbatch run_sims_given_param.sh $filename
  )
done
