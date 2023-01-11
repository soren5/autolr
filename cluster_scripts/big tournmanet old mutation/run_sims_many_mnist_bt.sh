#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --job-name=old_bt_sending_mni
#SBATCH --output=old_bt_sending_mni%j.log
export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models
for filename in ../../parameters/many_runs_no_elitism_big_tourn_old/mnist/*; 
do(
    [ -e "$filename" ] || continue 
    echo "sending: $filename"
    sbatch ../run_sims_given_param.sh $filename
  )
done