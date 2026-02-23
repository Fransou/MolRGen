#!/bin/bash
#SBATCH --job-name=sft_eval_molgen
#SBATCH --account=def-ibenayed
#SBATCH --time=04:00:00
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
CKPT_PATH=$3
SAVE_STEPS=$4
NUM_CHECKPOINTS=$5

# Calculate checkpoint steps if optional arguments are provided
CHECKPOINT_ARGS=""
if [[ -n "$SAVE_STEPS" && -n "$NUM_CHECKPOINTS" ]]; then
    START_INDEX=$((SLURM_ARRAY_TASK_ID * NUM_CHECKPOINTS))
    CHECKPOINT_LIST=""
    for ((i=0; i<NUM_CHECKPOINTS; i++)); do
        STEP_INDEX=$((START_INDEX + i + 1))
        STEP=$((STEP_INDEX * SAVE_STEPS))
        CHECKPOINT_LIST="$CHECKPOINT_LIST $STEP"
    done
    CHECKPOINT_ARGS="--subset $CHECKPOINT_LIST"
    echo "Evaluating steps: $CHECKPOINT_LIST"
fi

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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export docking_oracle=autodock_gpu
export scorer_exhaustiveness=4

ray start --head --node-ip-address 0.0.0.0 --dashboard-port=$DASHBOARD_PORT
ssh -N -f -R ${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT} $SLURM_JOB_USER@rorqual4

wandb offline
python -m openrlhf.cli.eval_batch_inference \
    --config $CONFIG \
    --ckpt_path $CKPT_PATH \
    --dashboard_port $DASHBOARD_PORT \
    $CHECKPOINT_ARGS
