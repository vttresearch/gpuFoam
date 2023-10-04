#!/bin/bash 
#SBATCH -J chemBench 
#SBATCH --output=runLog.out 
#SBATCH --error=runLog.err 
#SBATCH --time=00-24:30:00 
#SBATCH --mem-per-cpu=4096 
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=100
## #SBATCH --gres-flags=disable-binding
## #SBATCH --gres=gpu:2
## #SBATCH --gres=mps:100

#Partitions
#medium20
#medium24
#cuda
#all
#gen03_epyc 

#cd ${0%/*} || exit 1    # Run from this directory

module load openfoam/owngpu

export CUDA_VISIBLE_DEVICES=0,1

nvidia-cuda-mps-control -f > mps.log&

./Allrun

#Kill mps
kill %1


