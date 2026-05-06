#!/bin/bash
#SBATCH --job-name=meeko_preprocess
#SBATCH --account=def-ibenayed
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=192
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=sair_processed

cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzf $DATASET.tar.gz

cd $WORKING_DIR

export PATH=$HOME/autodock_vina_1_1_2_linux_x86/bin/vina:$PATH
export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

ray start --head --node-ip-address 0.0.0.0
python molrgen/data/meeko_process.py --data_path $DATA_PATH

cd $SLURM_TMPDIR
rm $DATASET/*.csv  # Remove original CSV files to save space
rm $DATASET/pdb_files/*.glg  # Remove intermediate files to save space
rm $DATASET/pdb_files/*.gpf  # Remove intermediate files to save space
rm $DATASET/pdb_files/*.box.pdb  # Remove intermediate files to save space

rm $DATASET/pdb_files/*.map  # Remove map files that will be computed at inf time
rm $DATASET/pdb_files/*.maps.fld  # Remove map files that will be computed at inf time
rm $DATASET/pdb_files/*.maps.xyz  # Remove map files that will be computed at inf time
rm $DATASET/pdb_files/*.pdbqt  # Remove pdbqt files that will be computed at inf time

mv $DATASET ${DATASET}_meeko

total_bytes=$(du -sb ${DATASET}_meeko | awk '{print $1}')
checkpoint_bytes=1000

tar -czf ${DATASET}_meeko.tar.gz ${DATASET}_meeko \
  --checkpoint=$checkpoint_bytes \
  --checkpoint-action=exec='
    bytes=$((TAR_CHECKPOINT * '"$checkpoint_bytes"'))
    if [ $bytes -gt '"$total_bytes"' ]; then bytes='"$total_bytes"'; fi
    kb=$((bytes / 1000))
    total_kb=$(( '"$total_bytes"' / 1000 ))
    percent=$((100 * bytes / '"$total_bytes"'))
    printf "%d/%d KB (%d%%)\r" $kb $total_kb $percent
  '


cp ${DATASET}_meeko.tar.gz $SCRATCH/MolGenData/
echo "Preprocessed data copied to $SCRATCH/MolGenData/${DATASET}_meeko.tar.gz"
