#!/bin/bash

module load miniconda/3.6
module load cuda/10.1

source ~/.bashrc

conda activate tf2_gpu


#PATH=/usr/bin/:/bin:/sbin:/usr/local/bin:/usr/sbin:/usr/local/slurm/slurm/:/usr/local/slurm/slurm/bin:/home/admin/scripts
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DATASET=$1
TYPE=$2
GPU_N=$3
Standard_N=$4
ll=$5


echo $DATASET $TYPE
#DATASET=kin40k; TYPE=reg; models=(nn solve swsgp titsias); splits=(0 1 2 3 4); for ((i=0; i<${#models[@]}; i+=1)); do for ((j=0; j<${#splits[@]}; j+=1)); do NUM_SPLIT=${$
models=(nn solve swsgp titsias);
splits=(0 1 2 3 4);

for ((i=0; i<${#models[@]}; i+=1));
do
    for ((j=0; j<${#splits[@]}; j+=1));
    do
    NUM_SPLIT=${splits[j]}
    Model=${models[i]}
    nip=1024
    if [ $Model = "nn" ]; then
        nip=50
    fi
    echo "python run_uci.py --dataset_name $DATASET --dataset_nsplit $NUM_SPLIT --modelSVGP $Model --Ptype $TYPE --scaling $Standard_N --ll $ll --nGPU $GPU_N --nip $nip --nhn1 50 --nhl1 1 --rdropout 0.5"   
    python run_uci.py --dataset_name $DATASET --dataset_nsplit $NUM_SPLIT --modelSVGP $Model --Ptype $TYPE --scaling $Standard_N --ll $ll --nGPU $GPU_N --nip $nip --nhn1 50 --nhl1 1 --rdropout 0.5
    done
done
wait