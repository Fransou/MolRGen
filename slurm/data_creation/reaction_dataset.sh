#!/bin/bash
#SBATCH --job-name=reac_dataset_creation
#SBATCH --account=def-ibenayed
#SBATCH --time=01:00:00
#SBATCH --mem=700G
#SBATCH --cpus-per-task=192
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-9

export WORKING_DIR=$HOME/MolGenDocking
DASHBOARD_PORT=$((8001 + SLURM_ARRAY_TASK_ID))

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate

cd $WORKING_DIR

ray start --head --dashboard-port=$DASHBOARD_PORT

ssh -N -f -R ${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT} $SLURM_JOB_USER@rorqual4

python molrgen/dataset/scripts/reaction_task/generate_reaction_dataset.py \
  -d data \
  -o data/synthesis/train_prompts_json_$SLURM_ARRAY_TASK_ID.jsonl \
  -n 50000 \
  --n_reaction_retry 200 --n_bb_retry 512
