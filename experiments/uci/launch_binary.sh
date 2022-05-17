#!/bin/bash

source ~/.bashrc

#PATH=/usr/bin/:/bin:/sbin:/usr/local/bin:/usr/sbin:/usr/local/slurm/slurm/:/usr/local/slurm/slurm/bin:/home/admin/scripts
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

datasets=(nomao miniboone crop_map ht_sensor bank_market default_or_credit magicg skinseg)
models=(nn solve swsgp titsias);
splits=(0 1 2 3 4);

for ((i=0; i<${#models[@]}; i+=1));
do
  for ((j=0; j<${#splits[@]}; j+=1));
  do
  	for ((k=0; k<${#datasets[@]}; k+=1));
  	do
  	  NUM_SPLIT=${splits[j]}
  	  Model=${models[i]}
	  DATASET=${datasets[k]}
  	  echo "python run_uci.py $DATASET $NUM_SPLIT $Model class MeanStd -1";
	  sbatch -p cccmd -A ada2_serv ./slurm_experiment.sh $DATASET $NUM_SPLIT $Model class MeanStd -1
  	done
  done
done
wait
echo all jobs are done!

