#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --job-name=sending_cif
#SBATCH --output=sending_ffc%j.log
export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models
for filename in ../../parameters/many_runs/fmni_from_cif/*; 
do(
    [ -e "$filename" ] || continue 
    echo "sending: $filename"
    sbatch ../run_sims_given_param.sh $filename
  )
done