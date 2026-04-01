#!/bin/bash
#SBATCH --job-name=batch_inference_molgen
#SBATCH --account=def-ibenayed
#SBATCH --time=00:00:00
#SBATCH --gpus=h100:4
#SBATCH --mem=248G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-0

DATASET=$1
CONFIG=$2

WORKING_DIR=$HOME/MolGenDocking
export DASHBOARD_PORT=$((8001 + SLURM_ARRAY_TASK_ID))
export DATA_PATH=$SLURM_TMPDIR/$DATASET

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz
cd $WORKING_DIR

cp data/properties.csv $SLURM_TMPDIR
ray start --head --node-ip-address 0.0.0.0 --dashboard-port=$DASHBOARD_PORT
ssh -N -f -R ${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT} $SLURM_JOB_USER@rorqual4

export PYTORCH_MULTIPROCESSING_START_METHOD=spawn

#export DEBUG_MODE=1
ray job submit \
   --address="http://127.0.0.1:$DASHBOARD_PORT" \
   --runtime-env-json='{"setup_commands": ["wandb offline"]}' \
   -- python3 -m openrlhf.cli.batch_inference \
   --config $CONFIG \
   --iter $SLURM_ARRAY_TASK_ID


export docking_oracle=autodock_gpu
export scorer_exhaustiveness=4

if [ "$DATASET" == "molgendata" ]; then
    python -m mol_gen_docking.score_completions \
      --iter $SLURM_ARRAY_TASK_ID \
      --input_file $CONFIG \
      --batch_size 1024 \
      --mol-generation
else
    python -m mol_gen_docking.score_completions \
      --iter $SLURM_ARRAY_TASK_ID \
      --input_file $CONFIG \
      --batch_size 1024
fi
