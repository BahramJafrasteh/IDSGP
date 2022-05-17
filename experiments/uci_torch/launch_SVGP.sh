#!/bin/bash

PATH=/usr/bin/:/bin:/sbin:/usr/local/bin:/usr/sbin:/usr/local/slurm/slurm/:/usr/local/slurm/slurm/bin:/home/admin/scripts
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
DATASET=$1

for split in {0..4}
do
        echo "sbatch -p cccmd -A ada2_serv $DIR/slurm_experiment.sh $DATASET $split titsias"
        sbatch -p cccmd -A ada2_serv $DIR/slurm_experiment.sh $DATASET $split titsias
done
