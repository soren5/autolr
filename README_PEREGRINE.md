# peregrine

Log into  `pg-gpu.hpc.rug.nl`, the interactive GPU node.

## modules

```bash
mkdir .local                    # place for local libraries
export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
```
### build models & test on interactive node

```bash
pip install -r requirements.txt
python -m utils.create_models
export CUDA_VISIBLE_DEVICES=0
python -m main
```

## test on GPU node

```bash
srun --gres=gpu:1 --partition=gpushort --time=01:00:00 --pty /bin/bash
python -m main
exit
```

## run on v100 GPU node

```bash
srun --gres=gpu:v100:1 --partition=gpushort --time=01:00:00 python3 -m main
```


## submit script v100 GPU node

```bash
#!/bin/bash
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:10:00

export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models
python3 -m main
```

Cheers,
Hanno
