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


echo $DATASET $TYPE
#DATASET=kin40k; TYPE=reg; models=(nn solve swgp titsias); splits=(0 1 2 3 4); for ((i=0; i<${#models[@]}; i+=1)); do for ((j=0; j<${#splits[@]}; j+=1)); do NUM_SPLIT=${splits[j]}; echo "$DATASET ${model$
models=(nn solve swsgp titsias);
splits=(0 1 2 3 4);

for ((i=0; i<${#models[@]}; i+=1));
do
  for ((j=0; j<${#splits[@]}; j+=1));
  do
    NUM_SPLIT=${splits[j]}
    Model=${models[i]}
    echo "python run_uci.py $DATASET $NUM_SPLIT $Model $TYPE $Standard_N $GPU_N";
    python run_uci.py $DATASET $NUM_SPLIT $Model $TYPE $Standard_N $GPU_N
  done
done
wait
echo all jobs are done!

