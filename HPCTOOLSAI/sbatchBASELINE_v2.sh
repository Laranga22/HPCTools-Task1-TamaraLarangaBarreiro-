#!/bin/bash
#SBATCH --job-name=baseline-torch  
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --ntasks-per-node=1        # Use 1 task (process) per node
#SBATCH -c 32         
#SBATCH --mem=32G
##SBATCH --gpus=1                   # Request 32 GB of memory 
#SBATCH --gres=gpu:a100:1          # Request 1 Nvidia A100 GPU
#SBATCH --time=00:30:00            # Set the total run time limit (HH:MM:SS)


# Activate your conda environment
source $STORE/mytorchdist/bin/deactivate

source $STORE/mytorchdist/bin/activate

# Run (BASELINE.py) with one GPU
which python
python lightningDDP_v2.py



