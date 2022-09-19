#!/bin/bash
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=autorl
#SBATCH --output=autorl_gpu_v100%j.log
export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models
python3 -m main --parameters parameters/adaptive_autolr_mutation_level.yml