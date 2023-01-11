#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=autorl_gpu
#SBATCH --output=autorl_gpu%j.log
python3 -m main --parameters ${1} --run ${2} 

