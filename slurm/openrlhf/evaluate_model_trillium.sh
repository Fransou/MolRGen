#!/bin/bash
#SBATCH --job-name=batch_inference_molgen
#SBATCH --account=def-ibenayed
#SBATCH --time=00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-0

DATASET=$1
CONFIG=$2

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


ray start --head --node-ip-address 0.0.0.0 --dashboard-port=$DASHBOARD_PORT
ssh -N -f -R ${DASHBOARD_PORT}:localhost:${DASHBOARD_PORT} $SLURM_JOB_USER@rorqual4


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

ray stop || true
