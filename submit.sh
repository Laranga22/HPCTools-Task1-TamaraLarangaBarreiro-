#!/bin/sh
#SBATCH --mem-per-cpu=100M
#SBATCH -t 00:10:00 # execution time hh:mm:ss *OB*
##SBATCH -N 1  #nodes (can be obtained from the two previous)
##SBATCH --ntasks-per-core ntasks # max ntasks per core
##SBATCH --ntasks-per-socket ntasks # max ntasks per socket
##SBATCH --ntasks-per-node ntasks # max ntasks per node
#SBATCH -p short

# The program(s) you want to execute:
srun ./dgesv 1024
