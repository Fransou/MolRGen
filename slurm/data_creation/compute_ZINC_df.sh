#!/bin/bash
#SBATCH --job-name=compute_ZINC_df
#SBATCH --account=rrg-josedolz
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking

cd $WORKING_DIR

export PATH=$HOME/qvina:$PATH
source $HOME/OpenRLHF/bin/activate

ray start --head --num-cpus 64

echo "Starting job on from $1 to $2"
python molrgen/compute_properties_ZINC.py \
  --batch-size 64 \
  --i-start $1 \
  --i-end $2
