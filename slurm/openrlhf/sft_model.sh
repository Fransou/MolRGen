#!/bin/bash
#SBATCH --job-name=sft_training_molgen
#SBATCH --account=def-ibenayed
#SBATCH --time=06:00:00
#SBATCH --gpus=h100:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

DATASET_PATH=$1
CONFIG=$2
OUT_PATH=$3

WORKING_DIR=$HOME/MolGenDocking

export DASHBOARD_PORT=9000

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate

cp $DATASET_PATH $SLURM_TMPDIR/sft_data.jsonl
cd $WORKING_DIR

ray start --head --node-ip-address 0.0.0.0 --dashboard-port=$DASHBOARD_PORT
ssh -N -f -R ${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT} $SLURM_JOB_USER@rorqual4


#export DEBUG_MODE=1
wandb offline
deepspeed --module openrlhf.cli.train_sft \
   --dataset $SLURM_TMPDIR/sft_data.jsonl \
   --save_path $OUT_PATH \
   --ckpt_path $OUT_PATH/ckpt \
   --config $CONFIG
