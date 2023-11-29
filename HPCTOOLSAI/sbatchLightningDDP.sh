#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -N 2
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8G
#SBATCH -c 32
#SBATCH --time=0-00:10:00

module purge
module load cesga/system miniconda3/22.11
eval "$(conda shell.bash hook)"
conda deactivate
source $STORE/mytorchdist/bin/deactivate
source $STORE/mytorchdist/bin/activate

# Set the number of GPUs and nodes
export WORLD_SIZE=4  # Total number of GPUs across all nodes
export NODE_RANK=$SLURM_NODEID
export RANK=$SLURM_PROCID

# debugging flags (optional)
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run with multiple GPUs using DDP
pythonint=$(which python)
srun python lightningDistTrainingDDP.py

