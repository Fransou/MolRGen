#!/bin/bash

source $HOME/.bashrc
export WORKING_DIR=$HOME/MolGenDocking
export DATASET=molgendata
export NCCL_ASYNC_ERROR_HANDLING=1

#cp $SCRATCH/MolGenData/$DATASET_prompts.tar.gz $SLURM_TMPDIR
#cd $SLURM_TMPDIR
#tar -xzf $DATASET.tar.gz
cd $WORKING_DIR

export DATA_PATH=$SLURM_TMPDIR/$DATASET
source $HOME/OpenRLHF/bin/activate

wandb offline
mkdir $SLURM_TMPDIR/reinvent_rpo_finetuned_model_$2_$3_$4_$5_$6_$7_$8_$9
HF_HUB_OFFLINE=1 python -m molrgen.baselines.reinvent.rl_opt_rpo \
  --output_dir $SLURM_TMPDIR/reinvent_rpo_finetuned_model_$2_$3_$4_$5_$6_$7_$8_$9 \
  --model_name $9 \
  --dataset $DATA_PATH/eval_data/eval_prompts.jsonl \
  --datasets-path $DATA_PATH \
  --batch_size $7 \
  --sigma $4 \
  --num_train_epochs 500 \
  --num_beams $5 \
  --train_on_beams $6 \
  --id_obj $2 \
  --rewards_to_pick $3 \
  --learning_rate $8 \
  --remote_rm_url http://$1:5001
