#!/bin/bash
#SBATCH --job-name=sair_processing
#SBATCH --account=def-ibenayed
#SBATCH --time=01:30:00
#SBATCH --mem=750G
#SBATCH --cpus-per-task=192
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=94

data_path=$SCRATCH/MolGenData/sair_data
files=($data_path/structures_compressed/*)

data_file=${files[$SLURM_ARRAY_TASK_ID]}


source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate
export WORKING_DIR=$HOME/MolGenDocking

cd $SLURM_TMPDIR
cp $data_file $SLURM_TMPDIR
mkdir sair_$SLURM_ARRAY_TASK_ID
tar -xzf $(basename $data_file) -C sair_$SLURM_ARRAY_TASK_ID
cp $data_path/sair.parquet $SLURM_TMPDIR/sair_$SLURM_ARRAY_TASK_ID

ray start --head --node-ip-address 0.0.0.0

cd $WORKING_DIR
python molrgen/data/SAIR_identify_pockets.py \
  --sair-dir $SLURM_TMPDIR/sair_$SLURM_ARRAY_TASK_ID \
  --output-dir $SLURM_TMPDIR/sair_pockets \
  --iou-threshold 0.6 \
  --topk 3

# Copy results back to SCRATCH
cd $SLURM_TMPDIR
cp sair_pockets/sair_pockets.csv $SCRATCH/MolGenData/sair_processed/sair_pockets_$SLURM_ARRAY_TASK_ID.csv
cp sair_pockets/pdb_files/* $SCRATCH/MolGenData/sair_processed/pdb_files/
