#!/bin/bash

n=$OMPI_COMM_WORLD_LOCAL_RANK
if [ `expr $n % 2` == 0 ]
then
    export CUDA_VISIBLE_DEVICES=0
    foamRun -solver multicomponentFluid -parallel
    #DEV=0
else
    export CUDA_VISIBLE_DEVICES=1
    foamRun -solver multicomponentFluid -parallel
	#DEV=1
fi

#numactl 


#echo $DEV
#export CUDA_VISIBLE_DEVICES=$DEV
#export CUDA_VISIBLE_DEVICES=0,1
#echo $DEV
#DEV=

#echo $N
#export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK






