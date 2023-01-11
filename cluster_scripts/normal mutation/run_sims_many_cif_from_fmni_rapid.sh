#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --job-name=sending_cif
#SBATCH --output=sending_cff%j.log
export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models
for filename in ../../parameters/many_runs/cif_from_fmni_rapid_transfer/*; 
do(
    [ -e "$filename" ] || continue 
    echo "sending: $filename"
    sbatch ../run_sims_given_param.sh $filename
  )
done