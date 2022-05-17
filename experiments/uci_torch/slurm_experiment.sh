#!/bin/bash
##################################################################
# Please note that you need to adapt this script to your job
# Submitting it as is will fail.
##################################################################
# Define the job name
#SBATCH --job-name=UCI
#
# Advised: your Email here, for job notification
#     (ALL = BEGIN, END, FAIL, REQUEUE)
#
# Set a pattern for the output file.
#SBATCH --output=log/salida_cluster_UCI-%j.out
#  By default both standard output and  standard  error are
# directed to a file of the name "slurm-%j.out", where the "%j"
# is replaced with the job allocation number.   The filename
# pattern may contain one or more replacement symbols, which are
# a percent sign "%" followed by a letter (e.g. %j).
#
# Supported replacement symbols are:
#     %j     Job allocation number.
#     %N     Main node name.
#
##################################################################
# The requested run-time
#
#SBATCH --time=5-0:0:0
# Acceptable time formats include "minutes", "minutes:seconds",
# "hours:minutes:seconds", "days-hours", "days-hours:minutes"
# and "days-hours:minutes:seconds"
#
# Slurm will kill your job after the requested period.
# The default time limit is the partition's time limit.
#
# Note that the lower the requested run-time, the higher the
# chances to get scheduled to 'fill in the gaps' between other
# jobs.
#
##################################################################
# Requested number of cores. Choose either of, or both of
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# Set a to be the number of process you want to launch and b the
# number of threads per process. Typically, for an MPI job with 8
# processes, set a=8, b=1. For an OpenMP job with 12 threads, set
# a=1, b=12. For a hybrid 8-process, 12-threads per process MPI
# job with OpenMP 'inside', set a=8, b=12.
#
##################################################################
# Launch job
#
# Note that the environment variables that were set when you
# submitted your job are copied and transmitted to the computing
# nodes. It nevertheless is good practice to reset them
# explicitly in the script.
#
# It is also good practice to use the environment variables such
# as $HOME, $TMP, etc rather than explicit paths to ensure smooth
# portability of the script.
#
# Select the setup you need and discard the rest.
#
### Simple sequential job
# If you have a simple non-parallel job, just launch it.
# So if it is called myprog.exe, just write :
#

source ~/.bashrc

conda activate torch
#conda activate tf2_gpu
#conda activate tf2

export CUDA_VISIBLE_DEVICES=1

DATASET=$1
SPLIT=$2
MODEL=$3

python -u run_uci.py $DATASET $SPLIT $MODEL
