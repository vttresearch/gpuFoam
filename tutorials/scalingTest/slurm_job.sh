#!/bin/bash 
#SBATCH -J bench 
#SBATCH --output=runLog.out 
#SBATCH --error=runLog.err 
#SBATCH --time=24:00:00 
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --account=project_2001715
#SBATCH --gres=gpu:a100:4


# to allocate gpus #SBATCH --gres=gpu:a100:4

source ~/OpenFOAM/OpenFOAM-dev/etc/bashrc

module load cuda

# Source run functions
. $WM_PROJECT_DIR/bin/tools/RunFunctions

nvidia-cuda-mps-control -f > mps.log&


./Allrun_gpu

