#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=def-ibenayed
#SBATCH --time=08:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100_3g.40gb:2
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=molgendata

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR
cp data/properties.csv $SLURM_TMPDIR

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

ray start --head --node-ip-address 0.0.0.0

pytest test/test_rewards/test_docking_API.py --accelerator=gpu

# Launch server
export docking_oracle=autodock_gpu
export scorer_exhaustiveness=4
uvicorn --host 0.0.0.0 --port 5001 molrgen.server:app &
sleep 10

pytest test/test_rewards/test_docking_server_autodock_gpu.py -x -s --accelerator gpu

kill -9 $(lsof -t -i :5001)
