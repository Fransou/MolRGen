#!/bin/bash
#SBATCH --job-name=batch_inference_molgen
#SBATCH --account=def-ibenayed
#SBATCH --time=00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-0


# $1: output
# $2: config

DATASET=molgendata

export RAY_TMPDIR=$SLURM_TMPDIR/ray
mkdir -p $RAY_TMPDIR

WORKING_DIR=$HOME/MolGenDocking
export DASHBOARD_PORT=$((8001 + SLURM_ARRAY_TASK_ID))
export DATA_PATH=$SLURM_TMPDIR/$DATASET

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate

ray stop || true

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz
cd $WORKING_DIR

cp data/properties.csv $SLURM_TMPDIR

export TRITON_CACHE_DIR=$SLURM_TMPDIR
export XDG_CONFIG_HOME=$SLURM_TMPDIR
export XDG_CACHE_HOME=$SLURM_TMPDIR
export HF_HOME=$SLURM_TMPDIR
export FLASHINFER_CACHE_DIR=$SLURM_TMPDIR/flashinfer_cache
export FLASHINFER_CUBIN_DIR=$SLURM_TMPDIR/flashinfer_cubin
export FLASHINFER_WORKSPACE_BASE=$SLURM_TMPDIR/flashinfer/workspace
export FLASHINFER_HOME=$SLURM_TMPDIR/flashinfer

ray start --head --node-ip-address 0.0.0.0 --dashboard-port=$DASHBOARD_PORT

BUFFER_TIME=1 PARSING_METHOD=none SERVER_MODE=batch python molrgen/server.py &
sleep 120
wandb offline

python molrgen/baselines/reinvent/rl_opt_rpo.py \
  --dataset $SCRATCH/MolGenData/$DATASET/test_data/test_prompts_ood.jsonl \
  --model_name $SCRATCH/Franso/Franso-reinvent_229M_256_prior \
  --output_dir $SCRATCH/MolGenOutput/reinvent/$1-$SLURM_TMPDIR \
  --id_obj $SLURM_ARRAY_TASK_ID

ray stop || true
