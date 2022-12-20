#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --job-name=bt_sending_loop
#SBATCH --output=bt_sending_loop%j.log
export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models

declare -a folders=("many_runs_no_elitism_big_tourn")
declare -a tasks=("cif" "fmni")

for folder in "${folders[@]}"
do(
  path=../parameters/$folder
  for task in "${tasks[@]}"
  do(
    path="$path"/$task
    # echo "files number: " ls "$path"/*.yml | wc -l
    for filename in "$path"/*.yml ; 
    do(
      echo "filename is:" $filename   
        for seed in $(seq 1 15)
        do(

          [ -e "$filename" ] || continue 
          echo "sending: $filename"
          sbatch run_sims_given_param_from_loop.sh $filename $seed
          )
        done
      ) 
    done
    )
  done
  )
done